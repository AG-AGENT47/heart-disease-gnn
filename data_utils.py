# data_utils.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from config import CONTINUOUS_INDICES, CATEGORICAL_INDICES, BIPARTITE_BINS


def load_data(filepath):
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    df = pd.read_csv(filepath, names=cols, na_values='?')
    df = df.dropna().reset_index(drop=True)
    y = (df['num'] > 0).astype(int).values
    X = df.drop('num', axis=1).values
    return X, y, cols[:-1]


def compute_gower_distance_matrix(X, ranges):
    n_samples, n_features = X.shape
    dist_matrix = np.zeros((n_samples, n_samples))
    for col_idx in CONTINUOUS_INDICES:
        col_data = X[:, col_idx].reshape(-1, 1)
        diff = np.abs(col_data - col_data.T)
        r = ranges[col_idx] if ranges[col_idx] > 0 else 1.0
        dist_matrix += (diff / r)
    for col_idx in CATEGORICAL_INDICES:
        col_data = X[:, col_idx].reshape(-1, 1)
        dist_matrix += (col_data != col_data.T).astype(float)
    dist_matrix /= n_features
    return dist_matrix


def build_psn_graph(X, y, k, train_ranges=None):
    # PSN uses scaled X for distance calculation, which is fine for Gower
    dist_mat = compute_gower_distance_matrix(X, train_ranges)
    adj_matrix = kneighbors_graph(dist_mat, n_neighbors=k, mode='connectivity', include_self=True, metric='precomputed')
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
    x_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return Data(x=x_tensor, edge_index=edge_index, y=y_tensor)


def build_bipartite_graph(X_raw, y, ranges, feature_names, X_scaled=None, bins=BIPARTITE_BINS):
    """
    X_raw: Used to determine Edges (Categories match perfectly)
    X_scaled: Used to populate Patient Node Features (Neural Net friendly)
    """
    if X_scaled is None: X_scaled = X_raw  # Fallback

    num_patients, num_features = X_raw.shape
    edges_src = []
    edges_dst = []
    current_attr_id = num_patients

    node_mapping = {}
    for i in range(num_patients):
        node_mapping[i] = f"Patient_{i}"

    for col_idx in range(num_features):
        col_data = X_raw[:, col_idx]  # USE RAW DATA FOR STRUCTURE
        feat_name = feature_names[col_idx]

        if col_idx in CONTINUOUS_INDICES:
            min_val = col_data.min()
            max_val = col_data.max()
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            binned_vals = np.digitize(col_data, bin_edges[:-1]) - 1

            for bin_i in range(bins):
                low = bin_edges[bin_i]
                high = bin_edges[bin_i + 1]
                node_id = current_attr_id + bin_i
                node_mapping[node_id] = f"{feat_name}: {low:.1f}-{high:.1f}"

            for p_idx, bin_val in enumerate(binned_vals):
                attr_node = current_attr_id + bin_val
                edges_src.append(p_idx)
                edges_dst.append(attr_node)
            current_attr_id += bins

        else:
            unique_vals = np.unique(col_data)
            max_cat = int(max(unique_vals))

            for val in unique_vals:
                node_id = current_attr_id + int(val)
                node_mapping[node_id] = f"{feat_name}: Cat_{int(val)}"

            for p_idx, val in enumerate(col_data):
                attr_node = current_attr_id + int(val)
                edges_src.append(p_idx)
                edges_dst.append(attr_node)
            current_attr_id += (max_cat + 1)

    num_attr_nodes = current_attr_id - num_patients

    # USE SCALED DATA FOR NODE FEATURES
    x_patients = torch.tensor(X_scaled, dtype=torch.float)
    x_attrs = torch.zeros((num_attr_nodes, num_features), dtype=torch.float)
    x_full = torch.cat([x_patients, x_attrs], dim=0)

    src_tensor = torch.tensor(edges_src + edges_dst, dtype=torch.long)
    dst_tensor = torch.tensor(edges_dst + edges_src, dtype=torch.long)
    edge_index = torch.stack([src_tensor, dst_tensor], dim=0)

    y_patients = torch.tensor(y, dtype=torch.long)
    y_attrs = torch.full((num_attr_nodes,), -1, dtype=torch.long)
    y_full = torch.cat([y_patients, y_attrs], dim=0)

    data = Data(x=x_full, edge_index=edge_index, y=y_full)
    data.num_patients = num_patients

    return data, node_mapping