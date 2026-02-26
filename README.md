# Heart Disease Risk Prediction via Bipartite Graph Neural Networks

A research project comparing **Patient Similarity Networks (PSN)** and **Bipartite Graph** representations with two learning paradigms — **GraphSAGE (GCN)** and **Node2Vec + XGBoost** — for heart disease risk prediction using the UCI Cleveland dataset.

**Live demo →** *(deploy on Streamlit Cloud — see below)*

---

## Overview

Traditional ML treats patients as independent samples. This project structures the problem as a graph:

| Graph Type | Idea |
|---|---|
| **PSN (Patient Similarity Network)** | Patients are nodes; edges connect the *k* most similar patients (Gower distance) |
| **Bipartite Graph** | Patients connect to shared *attribute nodes* (binned clinical features), so similar patients are linked transitively through shared risk factors |

Two models are compared on each graph:
- **GraphSAGE (Inductive GCN)** — end-to-end node classification
- **Node2Vec + XGBoost** — unsupervised graph embeddings fed to a gradient boosted classifier

All four combinations are evaluated under **strict 10-fold nested cross-validation** (inner 5-fold tuning) to avoid data leakage.

### Key Results

| Model | AUC-PR | Recall | Specificity | F1 | AUC-ROC |
|---|---|---|---|---|---|
| PSN + GCN | — | — | — | — | — |
| PSN + Node2Vec | — | — | — | — | — |
| **Bipartite + GCN** | **best** | — | — | — | — |
| Bipartite + Node2Vec | — | — | — | — | — |

*(Fill in your numbers from the training run output)*

---

## Project Structure

```
heart-disease-gnn/
├── streamlit_app.py          # Streamlit web app (Streamlit Cloud entry point)
├── models.py                 # InductiveGCN (GraphSAGE) definition
├── inference.py              # Bipartite graph expansion + GCN inference engine
├── config.py                 # All hyperparameters and dataset paths
├── data_utils.py             # PSN & Bipartite graph builders, data loading
├── requirements.txt          # Streamlit Cloud dependencies
│
├── artifacts/
│   ├── best_bipartite_model.pth   # Trained GraphSAGE weights (Bipartite, fold 1)
│   ├── test_graph_data.pt         # Reference graph used for inductive inference
│   └── node_mapping.pkl           # Attribute node ID → label mapping
│
├── data/
│   └── processed.cleveland.data   # UCI Cleveland Heart Disease dataset (303 patients)
│
├── plots/
│   ├── model_comparison_curves.png
│   ├── plot_roc_curve.png
│   ├── plot_pr_curve.png
│   ├── plot_confusion_matrix.png
│   ├── plot_learning_curve.png
│   ├── plot_bipartite_structure.png
│   ├── plot_psn_structure.png
│   └── feature_importance.png
│
└── training/                 # Full training pipeline (run locally)
    ├── main.py               # Nested CV training + model comparison
    ├── app.py                # FastAPI backend for local deployment
    ├── frontend.py           # Streamlit frontend for use with FastAPI backend
    ├── visualization.py      # ROC/PR/comparison plot generation
    └── ...
```

---

## Streamlit App (Live Demo)

The app takes 13 clinical inputs, dynamically expands the reference Bipartite Graph with the new patient as a node, and runs GCN inference.

### Deploy on Streamlit Cloud (free)

1. Fork this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, entry file `streamlit_app.py`
4. Click **Deploy** — no secrets or environment variables needed

### Run locally

```bash
git clone https://github.com/<your-username>/heart-disease-gnn
cd heart-disease-gnn
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Run the Full Training Pipeline

The training pipeline requires additional dependencies (Node2Vec, XGBoost, FastAPI):

```bash
pip install node2vec xgboost fastapi uvicorn torch-scatter torch-sparse torch-cluster networkx
cd training/
# Run from repo root so imports resolve correctly
cd ..
python training/main.py
```

This runs 10-fold nested CV and saves the best Bipartite GCN model to `artifacts/`.

### Local backend + frontend (alternative to Streamlit Cloud)

```bash
# Terminal 1 — start FastAPI backend
python training/app.py

# Terminal 2 — start Streamlit frontend
streamlit run training/frontend.py
```

---

## Dataset

**UCI Cleveland Heart Disease Dataset**
- 303 patients, 13 clinical features, binary label (disease present/absent)
- Features: age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, ST slope, major vessels, thalassemia

---

## Technical Details

**Graph construction:**
- *Bipartite*: 5 bins for continuous features (age, BP, cholesterol, heart rate, ST depression); categorical features map directly to attribute nodes
- *PSN*: Gower distance matrix → k-NN graph (k ∈ {5, 10, 15})

**Model:**
- 2-layer GraphSAGE with ReLU activations, dropout, and a linear classification head
- Trained with NLLLoss + Adam, early stopping (patience=15)
- Decision threshold optimised to 0.35 for clinical sensitivity

**Inference:**
- New patient is added inductively as a new node to the reference graph
- Edges drawn to matching attribute nodes based on feature values
- GNN runs forward pass; only the new patient's logit is read out

---

## Disclaimer

This is a research prototype built for the UW–Madison Network Biology course. It is **not intended for clinical use**.
