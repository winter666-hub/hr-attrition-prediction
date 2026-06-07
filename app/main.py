from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import os
from app.preprocess import transform_input

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
    prediction = model.predict_proba(df)[:, 1] >= 0.4
    return {"prediction": int(prediction[0])}