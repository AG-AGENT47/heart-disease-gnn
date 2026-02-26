# main.py
import os

# 1. Silence TensorFlow/System logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings

# 2. Silence Python Warnings (Clean Output)
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import pandas as pd
import torch
import networkx as nx
import itertools
import time
from node2vec import Node2Vec as N2V_Algo
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_curve, auc, f1_score,
                             roc_auc_score, accuracy_score, confusion_matrix,
                             recall_score, precision_score)
from xgboost import XGBClassifier

from config import *
from data_utils import load_data, build_psn_graph, build_bipartite_graph
from models import InductiveGCN


class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_gcn(train_data, val_data, params):
    input_dim = train_data.x.shape[1]
    model = InductiveGCN(in_channels=input_dim,
                         hidden_channels=params['hidden_channels'],
                         out_channels=2,
                         dropout=params['dropout'])

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()
    stopper = EarlyStopping(patience=15)

    for epoch in range(params['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(train_data)

        if hasattr(train_data, 'num_patients'):
            n_p = train_data.num_patients
            loss = criterion(out[:n_p], train_data.y[:n_p])
        else:
            loss = criterion(out, train_data.y)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(val_data)
            if hasattr(val_data, 'num_patients'):
                n_p_val = val_data.num_patients
                val_loss = criterion(out_val[:n_p_val], val_data.y[:n_p_val]).item()
            else:
                val_loss = criterion(out_val, val_data.y).item()

        stopper(val_loss, model)
        if stopper.early_stop:
            break

    if stopper.best_state:
        model.load_state_dict(stopper.best_state)
    return model


def train_node2vec_pipeline(train_graph, params):
    edge_index = train_graph.edge_index.numpy()
    G = nx.Graph()
    G.add_nodes_from(range(train_graph.num_nodes))
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)

    # Run Node2Vec (Quietly)
    n2v = N2V_Algo(G, dimensions=params['embedding_dim'], walk_length=params['walk_length'],
                   num_walks=params['walks_per_node'], p=params['p'], q=params['q'],
                   workers=1, quiet=True)
    model = n2v.fit(window=params['context_size'], min_count=1, batch_words=4)

    z = np.zeros((train_graph.num_nodes, params['embedding_dim']))
    for i in range(train_graph.num_nodes):
        if str(i) in model.wv:
            z[i] = model.wv[str(i)]
    return z


# --- TUNERS ---
def get_grid_combinations(grid_dict):
    keys = grid_dict.keys()
    values = (grid_dict[key] for key in keys)
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def tune_gcn(X_outer, y_outer, feature_names, graph_type, struct_grid):
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    struct_combos = get_grid_combinations(struct_grid)
    model_combos = get_grid_combinations(PARAM_GRID_GCN)

    best_score = -1
    best_struct = None
    best_model = None

    for s_params in struct_combos:
        for m_params in model_combos:
            val_scores = []
            for t_idx, v_idx in inner_cv.split(X_outer, y_outer):
                scaler = StandardScaler()
                X_t = scaler.fit_transform(X_outer[t_idx])
                X_v = scaler.transform(X_outer[v_idx])
                ranges = np.ptp(X_outer[t_idx], axis=0)
                y_t, y_v = y_outer[t_idx], y_outer[v_idx]

                if graph_type == 'PSN':
                    k = s_params['k']
                    g_t = build_psn_graph(X_t, y_t, k, ranges)
                    g_v = build_psn_graph(X_v, y_v, k, ranges)
                    mask = None
                else:
                    bins = s_params['bins']
                    g_t, _ = build_bipartite_graph(X_outer[t_idx], y_t, ranges, feature_names, X_scaled=X_t, bins=bins)
                    g_v, _ = build_bipartite_graph(X_outer[v_idx], y_v, ranges, feature_names, X_scaled=X_v, bins=bins)
                    mask = g_v.num_patients

                model = train_gcn(g_t, g_v, m_params)
                model.eval()
                with torch.no_grad():
                    out = model(g_v)
                    if mask: out = out[:mask]
                    probs = torch.exp(out)[:, 1].numpy()
                    p, r, _ = precision_recall_curve(y_v, probs)
                    val_scores.append(auc(r, p))

            avg = np.mean(val_scores)
            if avg > best_score:
                best_score = avg
                best_struct = s_params
                best_model = m_params

    return best_struct, best_model


def tune_n2v(X_outer, y_outer, feature_names, graph_type, struct_grid):
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    struct_combos = get_grid_combinations(struct_grid)
    n2v_combos = get_grid_combinations(PARAM_GRID_N2V)

    best_score = -1
    best_struct = None
    best_n2v = None

    for s_params in struct_combos:
        for n_params in n2v_combos:
            val_scores = []
            for t_idx, v_idx in inner_cv.split(X_outer, y_outer):
                scaler = StandardScaler()
                X_t_s = scaler.fit_transform(X_outer[t_idx])
                X_v_s = scaler.transform(X_outer[v_idx])
                ranges = np.ptp(X_outer[t_idx], axis=0)
                y_t, y_v = y_outer[t_idx], y_outer[v_idx]

                X_full = np.concatenate([X_t_s, X_v_s])
                if graph_type == 'BIPARTITE':
                    X_full_raw = np.concatenate([X_outer[t_idx], X_outer[v_idx]])

                if graph_type == 'PSN':
                    k = s_params['k']
                    g_full = build_psn_graph(X_full, np.concatenate([y_t, y_v]), k, ranges)
                else:
                    bins = s_params['bins']
                    g_full, _ = build_bipartite_graph(X_full_raw, np.concatenate([y_t, y_v]), ranges, feature_names,
                                                      X_scaled=None, bins=bins)

                z_full = train_node2vec_pipeline(g_full, n_params)
                z_patients = z_full[:len(y_t) + len(y_v)]

                # Silent Classifier
                clf = XGBClassifier(eval_metric='logloss', verbosity=0)
                clf.fit(z_patients[:len(y_t)], y_t)
                probs = clf.predict_proba(z_patients[len(y_t):])[:, 1]

                p, r, _ = precision_recall_curve(y_v, probs)
                val_scores.append(auc(r, p))

            avg = np.mean(val_scores)
            if avg > best_score:
                best_score = avg
                best_struct = s_params
                best_n2v = n_params

    return best_struct, best_n2v


def calculate_metrics(y_true, y_probs, y_preds):
    acc = accuracy_score(y_true, y_preds)
    rec = recall_score(y_true, y_preds)
    prec = precision_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds)
    try:
        roc = roc_auc_score(y_true, y_probs)
    except:
        roc = 0.5
    p, r, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(r, p)
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {'Accuracy': acc, 'Recall': rec, 'Specificity': spec, 'Precision': prec, 'F1': f1, 'AUC_ROC': roc,
            'AUC_PR': pr_auc}


def run_project():
    print("Loading Data...")
    X, y, feature_names = load_data(DATA_PATH)

    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    metrics_storage = {'PSN_GCN': [], 'PSN_N2V': [], 'BIPARTITE_GCN': [], 'BIPARTITE_N2V': []}

    print(f"Starting 4-Model Comparison (Outer: {OUTER_FOLDS}) with STRICT NESTED CV...")
    start_time = time.time()

    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        fold_start = time.time()
        print(f"\n--- Outer Fold {fold_idx + 1}/{OUTER_FOLDS} ---")
        X_tr, X_te = X[train_val_idx], X[test_idx]
        y_tr, y_te = y[train_val_idx], y[test_idx]

        # --- 1. TUNE ---
        print("   [Inner Loop] Tuning GCN Hyperparameters...")
        s_best_psn, m_best_psn = tune_gcn(X_tr, y_tr, feature_names, 'PSN', STRUCTURAL_GRID_PSN)
        s_best_bi, m_best_bi = tune_gcn(X_tr, y_tr, feature_names, 'BIPARTITE', STRUCTURAL_GRID_BI)
        print(f"     Best PSN: k={s_best_psn['k']}, {m_best_psn}")
        print(f"     Best Bi.: bins={s_best_bi['bins']}, {m_best_bi}")

        print("   [Inner Loop] Tuning Node2Vec Hyperparameters...")
        s_best_n2v_psn, n_best_psn = tune_n2v(X_tr, y_tr, feature_names, 'PSN', STRUCTURAL_GRID_PSN)
        s_best_n2v_bi, n_best_bi = tune_n2v(X_tr, y_tr, feature_names, 'BIPARTITE', STRUCTURAL_GRID_BI)
        print(f"     Best PSN N2V: k={s_best_n2v_psn['k']}, {n_best_psn}")
        print(f"     Best Bi. N2V: bins={s_best_n2v_bi['bins']}, {n_best_bi}")

        # --- 2. TRAIN & TEST ---
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        ranges = np.ptp(X_tr, axis=0)

        # PSN + GCN
        g_tr = build_psn_graph(X_tr_s, y_tr, s_best_psn['k'], ranges)
        g_te = build_psn_graph(X_te_s, y_te, s_best_psn['k'], ranges)
        model = train_gcn(g_tr, g_te, m_best_psn)
        model.eval()
        with torch.no_grad():
            logits = model(g_te)
            probs = torch.exp(logits)[:, 1].numpy()
            preds = logits.argmax(dim=1).numpy()
            metrics_storage['PSN_GCN'].append(calculate_metrics(y_te, probs, preds))

        # Bipartite + GCN
        g_tr, _ = build_bipartite_graph(X_tr, y_tr, ranges, feature_names, X_scaled=X_tr_s, bins=s_best_bi['bins'])
        g_te, mapping = build_bipartite_graph(X_te, y_te, ranges, feature_names, X_scaled=X_te_s,
                                              bins=s_best_bi['bins'])
        model = train_gcn(g_tr, g_te, m_best_bi)

        if fold_idx == 0:
            torch.save(model.state_dict(), 'best_bipartite_model.pth')
            torch.save(g_te, 'test_graph_data.pt')
            import pickle
            with open('node_mapping.pkl', 'wb') as f: pickle.dump(mapping, f)

        model.eval()
        with torch.no_grad():
            logits = model(g_te)[:g_te.num_patients]
            probs = torch.exp(logits)[:, 1].numpy()
            preds = logits.argmax(dim=1).numpy()
            metrics_storage['BIPARTITE_GCN'].append(calculate_metrics(y_te, probs, preds))

        # PSN + N2V
        X_full_s = np.concatenate([X_tr_s, X_te_s])
        X_full_raw = np.concatenate([X_tr, X_te])
        y_full = np.concatenate([y_tr, y_te])
        n_train = len(y_tr)

        g_psn_full = build_psn_graph(X_full_s, y_full, s_best_n2v_psn['k'], ranges)
        z_full = train_node2vec_pipeline(g_psn_full, n_best_psn)
        clf = XGBClassifier(eval_metric='logloss', verbosity=0)
        clf.fit(z_full[:n_train], y_tr)
        probs = clf.predict_proba(z_full[n_train:])[:, 1]
        preds = clf.predict(z_full[n_train:])
        metrics_storage['PSN_N2V'].append(calculate_metrics(y_te, probs, preds))

        # Bipartite + N2V
        g_bi_full, _ = build_bipartite_graph(X_full_raw, y_full, ranges, feature_names, X_scaled=None,
                                             bins=s_best_n2v_bi['bins'])
        z_bi_full = train_node2vec_pipeline(g_bi_full, n_best_bi)
        z_p = z_bi_full[:len(y_full)]
        clf = XGBClassifier(eval_metric='logloss', verbosity=0)
        clf.fit(z_p[:n_train], y_tr)
        probs = clf.predict_proba(z_p[n_train:])[:, 1]
        preds = clf.predict(z_p[n_train:])
        metrics_storage['BIPARTITE_N2V'].append(calculate_metrics(y_te, probs, preds))

        print(f"   Fold Time: {time.time() - fold_start:.1f}s")

    print("\n" + "=" * 85)
    print(f"{'Model':<15} | {'AUC-PR':<8} | {'Recall':<8} | {'Specif.':<8} | {'F1':<8} | {'Acc.':<8} | {'AUC-ROC':<8}")
    print("-" * 85)

    for model_name, metric_list in metrics_storage.items():
        df = pd.DataFrame(metric_list)
        avg = df.mean()
        print(
            f"{model_name:<15} | {avg['AUC_PR']:.3f}    | {avg['Recall']:.3f}    | {avg['Specificity']:.3f}    | {avg['F1']:.3f}    | {avg['Accuracy']:.3f}    | {avg['AUC_ROC']:.3f}")
    print("=" * 85)


if __name__ == "__main__":
    run_project()