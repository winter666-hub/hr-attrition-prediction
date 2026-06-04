from fastapi import FastAPI
import joblib

model = joblib.load("../models/catboost_best_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
app = FastAPI()

@app.get("/")
def root():
    return {"message": "HR Attrition Prediction API"}
