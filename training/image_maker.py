import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.patches as patches

# Set style for professional academic look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


# ==========================================
# FIGURE 1: RESULTS COMPARISON (Slide 6)
# ==========================================
def plot_results():
    models = ['PSN + GCN', 'PSN + N2V', 'Bipartite + GCN', 'Bipartite + N2V']
    auc_pr = [0.893, 0.884, 0.901, 0.859]
    auc_roc = [0.897, 0.877, 0.906, 0.874]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, auc_pr, width, label='AUC-PR', color='#4A90E2', alpha=0.9)
    rects2 = ax.bar(x + width / 2, auc_roc, width, label='AUC-ROC', color='#50E3C2', alpha=0.9)

    # Highlight the winner (Bipartite GCN) with a red border
    rects1[2].set_edgecolor('#C5050C')  # UW Madison Red-ish
    rects1[2].set_linewidth(3)
    rects2[2].set_edgecolor('#C5050C')
    rects2[2].set_linewidth(3)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison (10-Fold CV)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0.7, 0.95)
    ax.legend(loc='lower right')

    # Add labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()


# ==========================================
# FIGURE 2: EXPLAINABILITY (Slide 7)
# ==========================================
def plot_explainability():
    # Data from user logs
    factors = {
        'Patient\n(High Risk)': 0,  # Center
        'Age: 63-67': 0.9222,
        'Sex: Male (1)': 0.9145,
        'Major Vessels: 3': 0.8943,
        'Max HR: 96-108': 0.8722,
        'BP: 152-160': 0.8717
    }

    G = nx.Graph()
    center_node = 'Patient\n(High Risk)'

    for factor, weight in factors.items():
        if factor != center_node:
            G.add_edge(center_node, factor, weight=weight)

    pos = nx.spring_layout(G, k=0.5, seed=42)

    plt.figure(figsize=(8, 6))

    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[center_node], node_color='#C5050C', node_size=3000)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if n != center_node], node_color='#4A90E2',
                           node_size=2000)

    # Draw Edges (Thickness based on weight)
    weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.7)

    # Draw Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='white' if center_node else 'black')

    # Manually fix label colors (NetworkX is tricky with list of colors for labels)
    # Just overlaying text for clarity in the plot title/context
    plt.title(f"GNNExplainer: Top Factors for Patient Diagnosis\n(Edge Thickness = Importance Score)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ==========================================
# FIGURE 3: GRAPH TOPOLOGY (Slide 3)
# ==========================================
def plot_topology():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # PSN (Clique-ish)
    G_psn = nx.watts_strogatz_graph(8, 4, 0.3, seed=42)
    pos_psn = nx.circular_layout(G_psn)
    nx.draw(G_psn, pos_psn, ax=ax1, node_color='#4A90E2', node_size=500, with_labels=False, edge_color='gray')
    ax1.set_title("Method A: Patient Similarity Network (PSN)\nEdges = Statistical Similarity", fontsize=12,
                  fontweight='bold')

    # Bipartite
    B = nx.Graph()
    patients = [1, 2, 3, 4]
    attrs = ['Age', 'Sex', 'CP', 'BP']
    B.add_nodes_from(patients, bipartite=0)
    B.add_nodes_from(attrs, bipartite=1)
    B.add_edges_from([(1, 'Age'), (1, 'Sex'), (2, 'Age'), (2, 'CP'), (3, 'Sex'), (3, 'BP'), (4, 'CP'), (4, 'BP')])

    pos_bi = nx.bipartite_layout(B, patients)

    # Draw Patients
    nx.draw_networkx_nodes(B, pos_bi, nodelist=patients, node_color='#4A90E2', node_size=500, label='Patients', ax=ax2)
    # Draw Attrs
    nx.draw_networkx_nodes(B, pos_bi, nodelist=attrs, node_color='#F5A623', node_size=500, label='Attributes', ax=ax2)
    nx.draw_networkx_edges(B, pos_bi, ax=ax2, edge_color='gray')

    ax2.set_title("Method B: Bipartite Graph\nExplicit Patient-Attribute Links", fontsize=12, fontweight='bold')
    ax2.legend(['Patients', 'Attributes'])

    plt.tight_layout()
    plt.show()


# ==========================================
# FIGURE 4: NESTED CV SCHEME (Slide 5)
# ==========================================
def plot_nested_cv():
    fig, ax = plt.subplots(figsize=(12, 4))

    # Outer Loop
    ax.text(0, 1.5, "Outer Loop (Performance Est.)\n10 Folds", fontsize=12, fontweight='bold', va='center')
    for i in range(10):
        color = '#C5050C' if i == 9 else '#4A90E2'  # Highlight test fold
        rect = patches.Rectangle((i * 1.2 + 2, 1), 1, 1, linewidth=1, edgecolor='black', facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        if i == 9:
            ax.text(i * 1.2 + 2.5, 1.5, "Test", ha='center', va='center', color='white', fontweight='bold')

    # Arrow
    ax.annotate('', xy=(6, 0.8), xytext=(13, 1),
                arrowprops=dict(facecolor='black', shrink=0.05))

    # Inner Loop
    ax.text(0, 0, "Inner Loop (Hyperparam Tuning)\n5 Folds", fontsize=12, fontweight='bold', va='center')
    for i in range(5):
        color = '#F5A623' if i == 4 else '#4A90E2'  # Highlight val fold
        rect = patches.Rectangle((i * 1.2 + 4, -0.5), 1, 1, linewidth=1, edgecolor='black', facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        if i == 4:
            ax.text(i * 1.2 + 4.5, 0, "Val", ha='center', va='center', color='black', fontweight='bold')

    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 2.5)
    ax.axis('off')
    plt.title("Nested Cross-Validation Strategy", fontsize=16)
    plt.tight_layout()
    plt.show()


# Generate all
plot_results()
plot_explainability()
plot_topology()
plot_nested_cv()