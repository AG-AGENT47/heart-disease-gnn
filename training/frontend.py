import streamlit as st
import requests
import json

# Define the Backend URL (Make sure app.py is running!)
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

# --- Header ---
st.title("🫀 Heart Disease Risk Prediction")
st.markdown("""
This tool uses a **Bipartite Graph Neural Network (GNN)** to assess heart disease risk.
Please enter the patient's clinical details below.
""")

st.markdown("---")

# --- Input Form ---
# We use columns to make the layout look professional
col1, col2 = st.columns(2)

with col1:
    st.header("Patient Demographics")
    age = st.slider("Age", 20, 90, 60)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

    st.header("Vitals")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[1, 0],
                       format_func=lambda x: "True" if x == 1 else "False")

with col2:
    st.header("Cardiac Signs")
    cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4],
                      format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain",
                                             4: "Asymptomatic"}.get(x))

    exang = st.selectbox("Exercise Induced Angina?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                           format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality",
                                                  2: "Left Ventricular Hypertrophy"}.get(x))

    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST", options=[1, 2, 3],
                         format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}.get(x))

    ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.selectbox("Thalassemia", options=[3, 6, 7],
                        format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}.get(x))

# --- Prediction Logic ---
st.markdown("---")
if st.button("Analyze Risk", type="primary", use_container_width=True):
    # Construct the payload matching the Pydantic schema in app.py
    payload = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    with st.spinner("Constructing Graph & Querying GCN Model..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()

                # Display Results
                prediction = result['prediction']
                prob = result['probability']
                percent = result['risk_percent']

                if prediction == "High Risk":
                    st.error(f"### Result: {prediction}")
                    st.write(f"The model predicts a **{percent}** probability of heart disease.")
                    st.progress(prob)
                else:
                    st.success(f"### Result: {prediction}")
                    st.write(f"The model predicts a **{percent}** probability of heart disease.")
                    st.progress(prob)

                st.info(
                    f"Note: This prediction used a clinically adjusted sensitivity threshold of **{result['threshold_used']}**.")

            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend. Is 'app.py' running? Error: {e}")