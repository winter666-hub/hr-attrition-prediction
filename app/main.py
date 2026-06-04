from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

model = joblib.load("../models/catboost_best_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
app = FastAPI()


class EmployeeData(BaseModel):
    Age: int
    MonthlyIncome: int
    DistanceFromHome: int
    TotalWorkingYears: int

@app.get("/")
def root():
    return {"message": "HR Attrition Prediction API"}

@app.post("/predict")
def predict(data: EmployeeData)