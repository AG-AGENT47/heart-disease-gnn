# visualize.py
import torch
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_networkx

from config import *
from data_utils import load_data, build_psn_graph, build_bipartite_graph
from models import InductiveGCN

# Set style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


def plot_comparative_results():
    """Generates the bar chart with error bars for the 4 models."""
    # Data from your final run (Run 5)
    models = ['Bipartite + GCN', 'PSN + GCN', 'PSN + Node2Vec', 'Bipartite + Node2Vec']
    means = [0.8999, 0.8886, 0.8585, 0.8442]
    stds = [0.0612, 0.0857, 0.0823, 0.1173]

    x_pos = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.8, ecolor='black', capsize=10)

    # Color the winner green
    bars[0].set_color('#2ca02c')  # Green
    bars[1].set_color('#1f77b4')  # Blue
    bars[2].set_color('#ff7f0e')  # Orange
    bars[3].set_color('#d62728')  # Red

    ax.set_ylabel('AUC-PR Score')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.set_title('Final Model Comparison: AUC-PR (Mean ± Std Dev)')
    ax.set_ylim(0.7, 1.0)

    # Add text labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., 1.01 * height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('plot_comparison.png', dpi=300)
    print("Saved plot_comparison.png")


def plot_graph_structure(X_scaled, y, feature_names, ranges):
    """Visualizes subgraphs of PSN and Bipartite graphs."""

    # --- 1. PSN Visualization ---
    print("Generating PSN Visualization...")
    g_psn = build_psn_graph(X_scaled, y, k=5, train_ranges=ranges)

    # Convert to NetworkX
    G = to_networkx(g_psn, to_undirected=True)

    # Subsample nodes for clarity (Take first 40 nodes)
    subset = range(40)
    G_sub = G.subgraph(subset)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_sub, seed=42)

    # Color by Label
    node_colors = ['#ff9999' if y[i] == 1 else '#66b3ff' for i in subset]

    nx.draw_networkx_nodes(G_sub, pos, node_size=300, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.3)
    nx.draw_networkx_labels(G_sub, pos, font_size=8)

    # Custom Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Disease', markerfacecolor='#ff9999', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Healthy', markerfacecolor='#66b3ff', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("Patient Similarity Network (Subgraph k=5)")
    plt.axis('off')
    plt.savefig('plot_psn_structure.png', dpi=300)
    print("Saved plot_psn_structure.png")

    # --- 2. Bipartite Visualization ---
    print("Generating Bipartite Visualization...")
    g_bi, _ = build_bipartite_graph(X, y, ranges, feature_names, X_scaled=X_scaled)

    # Manually build NX for Bipartite to handle types
    G_bi = nx.Graph()

    # Edges
    edge_index = g_bi.edge_index.numpy()
    edges = list(zip(edge_index[0], edge_index[1]))

    # Filter for subgraph: Patients 0-10 and their connected attributes
    p_subset = set(range(303))  # Plot everyone!

    a_subset = set()
    valid_edges = []

    for u, v in edges:
        if u in p_subset:
            a_subset.add(v)
            valid_edges.append((u, v))
        elif v in p_subset:
            a_subset.add(u)
            valid_edges.append((u, v))

    G_bi.add_nodes_from(list(p_subset), type='patient')
    G_bi.add_nodes_from(list(a_subset), type='attribute')
    G_bi.add_edges_from(valid_edges)

    plt.figure(figsize=(20, 20))  # Make the canvas huge

    pos = nx.spring_layout(G_bi, k=0.3, seed=42)

    # Colors
    colors = []
    for n in G_bi.nodes():
        if n < len(y):  # Patient
            colors.append('#ff9999' if y[n] == 1 else '#66b3ff')
        else:  # Attribute
            colors.append('#90ee90')  # Light Green

    nx.draw_networkx_nodes(G_bi, pos, node_size=300, node_color=colors, alpha=0.9)
    nx.draw_networkx_edges(G_bi, pos, alpha=0.3)

    # Labels (Only for Attributes to reduce clutter)
    labels = {n: '' if n < len(y) else 'Attr' for n in G_bi.nodes()}
    # nx.draw_networkx_labels(G_bi, pos, labels=labels, font_size=8)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Patient (Disease)', markerfacecolor='#ff9999',
                   markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Patient (Healthy)', markerfacecolor='#66b3ff',
                   markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Attribute Hub', markerfacecolor='#90ee90', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("Bipartite Patient-Attribute Graph (Subgraph)")
    plt.axis('off')
    plt.savefig('plot_bipartite_structure.png', dpi=300)
    print("Saved plot_bipartite_structure.png")


def plot_champion_performance():
    """Loads the saved model and plots CM, ROC, PR."""
    print("Generating Champion Model Plots...")

    # Load Artifacts
    try:
        # Fix for newer pytorch
        data = torch.load('test_graph_data.pt', weights_only=False)
        model_state = torch.load('best_bipartite_model.pth', weights_only=False)
    except:
        print("Could not load saved model. Did you run main.py?")
        return

    # Init Model
    model = InductiveGCN(in_channels=13, hidden_channels=16, out_channels=2, dropout=0.5)
    model.load_state_dict(model_state)
    model.eval()

    # Inference
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        # Mask for patients only
        if hasattr(data, 'num_patients'):
            logits = logits[:data.num_patients]
            y_true = data.y[:data.num_patients].numpy()
        else:
            y_true = data.y.numpy()

        probs = torch.exp(logits)[:, 1].numpy()
        preds = logits.argmax(dim=1).numpy()

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Healthy', 'Disease'], yticklabels=['Healthy', 'Disease'])
    plt.title('Confusion Matrix (Champion Model)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('plot_confusion_matrix.png', dpi=300)
    print("Saved plot_confusion_matrix.png")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('plot_roc_curve.png', dpi=300)
    print("Saved plot_roc_curve.png")

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('plot_pr_curve.png', dpi=300)
    print("Saved plot_pr_curve.png")


def generate_learning_curve(X, y, feature_names, ranges):
    """Runs a quick demo training to generate a loss curve."""
    print("Generating Learning Curve...")

    # Quick Split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Build Graph
    ranges = np.ptp(X_train, axis=0)
    g_train, _ = build_bipartite_graph(X_train, y_train, ranges, feature_names, X_scaled=X_train_s)

    # Train
    model = InductiveGCN(in_channels=13, hidden_channels=16, out_channels=2, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()

    losses = []
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(g_train)
        loss = criterion(out[:g_train.num_patients], g_train.y[:g_train.num_patients])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('NLL Loss')
    plt.title('Learning Curve (Bipartite GCN)')
    plt.legend()
    plt.savefig('plot_learning_curve.png', dpi=300)
    print("Saved plot_learning_curve.png")


if __name__ == "__main__":
    # Load Data
    X, y, feature_names = load_data(DATA_PATH)
    ranges = np.ptp(X, axis=0)

    # Scale for visualization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run Plotters
    plot_comparative_results()
    plot_graph_structure(X_scaled, y, feature_names, ranges)
    plot_champion_performance()
    generate_learning_curve(X, y, feature_names, ranges)

    print("\nAll visualizations complete!")