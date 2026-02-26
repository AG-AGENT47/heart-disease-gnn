# explain.py
import torch
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer
from models import InductiveGCN

# 1. Load Artifacts
print("Loading model and graph...")
# Load Graph Data
data = torch.load('test_graph_data.pt', weights_only=False)
# Load Mapping
with open('node_mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)

# 2. Load Model Architecture & Weights
# (Must match config used in main.py)
model = InductiveGCN(in_channels=13, hidden_channels=16, out_channels=2, dropout=0.5)
model.load_state_dict(torch.load('best_bipartite_model.pth', weights_only=False))
model.eval()

# 3. Select a Patient to Explain
# Let's find a patient who actually has Heart Disease (Label 1)
sick_patient_indices = (data.y[:data.num_patients] == 1).nonzero(as_tuple=True)[0]
target_node_idx = sick_patient_indices[0].item()  # Pick the first sick patient

print(f"\nExplaining Prediction for Node {target_node_idx} ({mapping[target_node_idx]})...")

# 4. Initialize GNNExplainer
# It learns a mask over edges to see which edges are most important for the prediction
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

# 5. Run Explanation
explanation = explainer(data.x, data.edge_index, index=target_node_idx)

# 6. Interpret Results
# The explainer gives us an 'edge_mask' (importance 0 to 1 for each edge)
edge_mask = explanation.edge_mask
edge_index = data.edge_index

# Find edges connected to our target node
src, dst = edge_index
connected_mask = (src == target_node_idx) | (dst == target_node_idx)

# Filter for edges that are BOTH connected to our patient AND important
relevant_indices = connected_mask.nonzero(as_tuple=True)[0]
important_scores = edge_mask[relevant_indices]

# Sort by importance
sorted_indices = relevant_indices[important_scores.argsort(descending=True)]

print(f"\nTop Factors contributing to High Risk for Patient {target_node_idx}:")
print("-" * 50)

top_k = 5
count = 0
seen_factors = set()

for idx in sorted_indices:
    if count >= top_k: break

    # Who is the neighbor?
    # If src is target, neighbor is dst.
    s, d = src[idx].item(), dst[idx].item()
    neighbor_id = d if s == target_node_idx else s

    # Look up meaning
    factor_name = mapping.get(neighbor_id, f"Node_{neighbor_id}")
    score = edge_mask[idx].item()

    # Dedup (graph is undirected, might appear twice)
    if factor_name in seen_factors: continue
    seen_factors.add(factor_name)

    print(f"Factor: {factor_name:<30} | Importance: {score:.4f}")
    count += 1

print("-" * 50)
print("These are the clinical attributes (or similar patients) the model")
print("relied on most to make this diagnosis.")