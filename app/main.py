from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import os
from app.preprocess import transform_input
from app.database import SessionLocal
from app.models import Predictions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "catboost_best_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
feature_cols = scaler.feature_names_in_
app = FastAPI()


class EmployeeData(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EmployeeNumber: int
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    
    

@app.get("/")
def root():
    return {"message": "HR Attrition Prediction API"}

@app.post("/predict")
def predict(data: EmployeeData):
    input_dict = data.model_dump()
    df = transform_input(input_dict, scaler, feature_cols)
    prob = model.predict_proba(df)[:, 1][0]
    prediction = int(prob >= 0.4)

    # DB 저장
    db = SessionLocal()
    try:
        record = Predictions(
            employee_number=data.EmployeeNumber,
            prediction=prediction,
            probability=float(prob)
        )
        db.add(record)
        db.commit()
    finally:
        db.close()

    return {"prediction": prediction, "probability": round(float(prob), 4)}