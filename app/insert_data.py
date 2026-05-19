import pandas as pd

from database import SessionLocal
from models import Employees

df = pd.read_csv("../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

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
        stock_option_level = row["StockOptionLevel"],
        # 경력
        job_level = row["JobLevel"],
        total_working_years = row["TotalWorkingYears"],
        num_companies_worked = row["NumCompaniesWorked"],
        years_at_company = row["YearsAtCompany"],
        years_in_current_role = row["YearsInCurrentRole"],
        years_since_last_promotion = row["YearsSinceLastPromotion"],
        years_with_curr_manager = row["YearsWithCurrManager"],
        training_times_last_year = row["TrainingTimesLastYear"],
        # 직무
        job_role = row["JobRole"],
        department = row["Department"],
        # 근무 환경
        business_travel = row["BusinessTravel"],
        distance_from_home = row["DistanceFromHome"],
        over_time = row["OverTime"] == "Yes",
        # 만족도
        environment_satisfaction = row["EnvironmentSatisfaction"],
        job_satisfaction = row["JobSatisfaction"],
        relationship_satisfaction = row["RelationshipSatisfaction"],
        job_involvement = row["JobInvolvement"],
        performance_rating = row["PerformanceRating"],
        work_life_balance = row["WorkLifeBalance"],
        # target
        attrition = row["Attrition"] == "Yes"
    )

    session.add(employee)

session.commit()
session.close()
