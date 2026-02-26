# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from inference import HeartDiseasePredictor
import uvicorn

# 1. Define Input Schema (Data Validation)
class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# 2. Initialize App & Model
app = FastAPI(title="Heart Disease Graph Prediction API")
predictor = HeartDiseasePredictor()

@app.get("/")
def home():
    return {"message": "Heart Disease GCN API is running."}

@app.post("/predict")
def predict_heart_disease(patient: PatientData):
    """
    Accepts patient data, updates the Bipartite Graph dynamically,
    and returns a risk assessment.
    """
    data_dict = patient.dict()
    result = predictor.predict(data_dict, threshold=0.35)
    return result

if __name__ == "__main__":
    print("Starting Web Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)