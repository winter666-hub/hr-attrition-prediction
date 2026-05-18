import pandas as pd

from database import SessionLocal
from models import Employees

df = pd.read_csv("../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(df.head())

print(df.columns)

session = SessionLocal()

for index, row in df.iterrows():
    employee = Employees(
        # 개인정보
        employee_number = row["EmployeeNumber"],
        age = row["Age"],
        gender = row["Gender"],
        marital_status = row["MaritalStatus"],
        # 학력
        education = row["Education"],
        education_field = row["EducationField"],
        # 급여
        daily_rate = row["DailyRate"],
        hourly_rate = row["HourlyRate"],
        monthly_income = row["MonthlyIncome"],
        monthly_rate = row["MonthlyRate"],
        percent_salary_hike = row["PercentSalaryHike"],
        stock_option_level = row["StockOptionLevel"]
    )