# inference.py
import torch
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from config import DATA_PATH, CONTINUOUS_INDICES, BIPARTITE_BINS
from data_utils import load_data
from models import InductiveGCN
from torch_geometric.data import Data


class HeartDiseasePredictor:
    def __init__(self):
        print("Initializing Inference Engine...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Load Reference Data & Re-fit Scaler
        print("Loading raw data to establish feature scaling...")
        X_raw, _, self.feature_names = load_data(DATA_PATH)
        self.scaler = StandardScaler()
        self.scaler.fit(X_raw)
        self.ranges = np.ptp(X_raw, axis=0)
        self.X_raw_ref = X_raw  # Keep for binning structure

        # 2. Load the Reference Graph & Mapping
        print("Loading Graph Artifacts...")
        # FIX: Added weights_only=False to allow loading the custom Graph Object
        self.base_graph = torch.load('artifacts/test_graph_data.pt', map_location=self.device, weights_only=False)

        with open('artifacts/node_mapping.pkl', 'rb') as f:
            self.node_mapping = pickle.load(f)

        # 3. Load Model
        print("Loading Model...")
        # Note: Must match the hidden_channels used in training (likely 16 or 32)
        self.model = InductiveGCN(in_channels=13, hidden_channels=16, out_channels=2, dropout=0)

        # FIX: Added weights_only=False (good practice for local files)
        self.model.load_state_dict(torch.load('artifacts/best_bipartite_model.pth', map_location=self.device, weights_only=False))

        self.model.to(self.device)
        self.model.eval()

    def _get_edges_for_patient(self, patient_features_raw, start_node_idx):
        """
        Determines which Attribute Nodes the new patient connects to.
        """
        edges_dst = []

        # We need to reconstruct the attribute node IDs based on the reference data
        current_attr_id = self.base_graph.num_patients  # Start after the existing patients

        for col_idx, val in enumerate(patient_features_raw):
            col_data = self.X_raw_ref[:, col_idx]

            if col_idx in CONTINUOUS_INDICES:
                # Re-calculate bins exactly as training did
                min_val, max_val = col_data.min(), col_data.max()
                bin_edges = np.linspace(min_val, max_val, BIPARTITE_BINS + 1)

                # Find which bin this new patient falls into
                bin_idx = np.digitize(val, bin_edges[:-1]) - 1
                bin_idx = max(0, min(bin_idx, BIPARTITE_BINS - 1))  # Clip safety

                target_node = current_attr_id + bin_idx
                edges_dst.append(target_node)
                current_attr_id += BIPARTITE_BINS

            else:
                # Categorical
                unique_vals = np.unique(col_data)
                max_cat = int(max(unique_vals))

                if val in unique_vals:
                    target_node = current_attr_id + int(val)
                    edges_dst.append(target_node)

                current_attr_id += (max_cat + 1)

        return edges_dst

    def predict(self, input_dict, threshold=0.35):
        # 1. Convert Dict to Array (Ordered)
        feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

        try:
            raw_vec = np.array([float(input_dict[f]) for f in feature_order]).reshape(1, -1)
        except KeyError as e:
            return {"error": f"Missing feature: {e}"}

        # 2. Scale Features
        scaled_vec = self.scaler.transform(raw_vec)
        x_new_patient = torch.tensor(scaled_vec, dtype=torch.float).to(self.device)

        # 3. Create Dynamic Graph (Base Graph + 1 New Node)
        # New Node Index
        new_node_idx = self.base_graph.x.shape[0]

        # Add new features to X
        x_full = torch.cat([self.base_graph.x, x_new_patient], dim=0)

        # Calculate Edges for new patient
        dst_nodes = self._get_edges_for_patient(raw_vec[0], new_node_idx)

        new_src = torch.tensor([new_node_idx] * len(dst_nodes), dtype=torch.long)
        new_dst = torch.tensor(dst_nodes, dtype=torch.long)

        edge_index_new = torch.stack([
            torch.cat([new_src, new_dst]),
            torch.cat([new_dst, new_src])
        ], dim=0).to(self.device)

        edge_index_full = torch.cat([self.base_graph.edge_index, edge_index_new], dim=1)

        # 4. Predict
        with torch.no_grad():
            out = self.model(x_full, edge_index_full)
            logits = out[-1].unsqueeze(0)
            probs = torch.exp(logits)[:, 1].item()
            pred_class = 1 if probs >= threshold else 0

        return {
            "prediction": "High Risk" if pred_class == 1 else "Low Risk",
            "probability": round(probs, 4),
            "threshold_used": threshold,
            "risk_percent": f"{round(probs * 100, 2)}%"
        }