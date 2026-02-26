# config.py

# --- Dataset Paths ---
DATA_PATH = 'data/processed.cleveland.data'

# --- Feature Types (Indices based on processed dataset) ---
# 0:age, 1:sex, 2:cp, 3:trestbps, 4:chol, 5:fbs, 6:restecg,
# 7:thalach, 8:exang, 9:oldpeak, 10:slope, 11:ca, 12:thal
CONTINUOUS_INDICES = [0, 3, 4, 7, 9]
CATEGORICAL_INDICES = [1, 2, 5, 6, 8, 10, 11, 12]

# --- Validation Strategy ---
OUTER_FOLDS = 10
INNER_FOLDS = 5
RANDOM_SEED = 42

# --- FIXED PARAMETERS ---
BIPARTITE_BINS = 5  # <--- Crucial for data_utils.py

# --- 1. GCN Model Hyperparameters ---
PARAM_GRID_GCN = {
    'hidden_channels': [16, 32],
    'dropout': [0.3, 0.5],
    'lr': [0.01],
    'epochs': [200] # Controlled by Early Stopping
}

# --- 2. Structural Hyperparameters (Graph Topology) ---
# For PSN: 'k' controls neighborhood size
STRUCTURAL_GRID_PSN = {
    'k': [5, 10, 15]
}

# For Bipartite: 'bins' controls granularity of continuous features
STRUCTURAL_GRID_BI = {
    'bins': [3, 5, 7]
}

# --- 3. Node2Vec Hyperparameters ---
# Tuning p and q allows N2V to alternate between BFS (homophily) and DFS (structure)
PARAM_GRID_N2V = {
    'embedding_dim': [64],
    'walk_length': [20],
    'walks_per_node': [10],
    'context_size': [10],
    'p': [0.5, 1, 2],  # Tuning bias
    'q': [0.5, 1, 2]   # Tuning bias
}

# Default Params for Node2Vec (Baseline fallback)
NODE2VEC_PARAMS = {
    'embedding_dim': 64,
    'walk_length': 20,
    'context_size': 10,
    'walks_per_node': 10,
    'p': 1,
    'q': 1
}