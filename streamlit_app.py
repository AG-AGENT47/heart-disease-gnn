import streamlit as st

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

import sys
import os

# Ensure the repo root is on the path so inference.py can import its siblings
sys.path.insert(0, os.path.dirname(__file__))

from inference import HeartDiseasePredictor


@st.cache_resource(show_spinner="Loading GNN model and graph artifacts...")
def load_predictor():
    return HeartDiseasePredictor()


# --- Header ---
st.title("🫀 Heart Disease Risk Prediction")
st.markdown(
    "This tool uses a **Bipartite Graph Neural Network (GNN)** trained on the "
    "[UCI Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). "
    "Enter the patient's clinical details below to get a risk assessment."
)
st.markdown("---")

# --- Load model (cached) ---
try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# --- Input Form ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Demographics")
    age = st.slider("Age", 20, 90, 55)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

    st.subheader("Vitals")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl?",
        options=[0, 1],
        format_func=lambda x: "True" if x == 1 else "False"
    )

with col2:
    st.subheader("Cardiac Signs")
    cp = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Typical Angina",
            2: "Atypical Angina",
            3: "Non-anginal Pain",
            4: "Asymptomatic"
        }.get(x)
    )
    exang = st.selectbox(
        "Exercise Induced Angina?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    restecg = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }.get(x)
    )
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, step=0.1)
    slope = st.selectbox(
        "Slope of Peak Exercise ST",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}.get(x)
    )
    ca = st.slider("Number of Major Vessels (0–3)", 0, 3, 0)
    thal = st.selectbox(
        "Thalassemia",
        options=[3, 6, 7],
        format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}.get(x)
    )

# --- Predict ---
st.markdown("---")
if st.button("Analyze Risk", type="primary", use_container_width=True):
    payload = {
        "age": age, "sex": sex, "cp": cp,
        "trestbps": trestbps, "chol": chol, "fbs": fbs,
        "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    with st.spinner("Expanding bipartite graph and running GNN inference..."):
        result = predictor.predict(payload, threshold=0.35)

    if "error" in result:
        st.error(f"Prediction error: {result['error']}")
    else:
        st.markdown("### Result")
        prob = result["probability"]
        if result["prediction"] == "High Risk":
            st.error(f"**High Risk Detected** — {result['risk_percent']} probability of disease")
        else:
            st.success(f"**Low Risk** — {result['risk_percent']} probability of disease")

        st.progress(prob)
        st.caption(
            f"Clinically-adjusted decision threshold: **{result['threshold_used']}** "
            f"(optimised for sensitivity on the Cleveland dataset)"
        )

# --- Info expander ---
with st.expander("How does this work?"):
    st.markdown("""
**Architecture:** Inductive GraphSAGE (2-layer GCN)
**Graph type:** Bipartite — patients connect to shared *attribute nodes* (binned clinical features),
enabling the model to propagate similarity through shared risk factors rather than raw feature distance.

**Training:** 10-fold nested cross-validation on the
[UCI Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (303 patients, 13 features).

**Inference:** Your inputs are added as a new node to the reference graph.
Edges are drawn to the matching attribute nodes, and the GNN predicts risk by message-passing through those connections.

⚠️ *This is a research prototype and is not intended for clinical use.*
""")
