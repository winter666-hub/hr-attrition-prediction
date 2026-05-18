from sqlalchemy import Column, Integer, String, Boolean
from base import Base

class Employees(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True)
    # 개인정보
    employee_number = Column(Integer, unique=True)
    age = Column(Integer)
    gender = Column(String)     
    marital_status = Column(String)              # 결혼 여부
    # 학력
    education = Column(Integer) # 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor
    education_field = Column(String)

    # 급여
    daily_rate = Column(Integer)
    hourly_rate = Column(Integer)
    monthly_income = Column(Integer)
    monthly_rate = Column(Integer)
    percent_salary_hike = Column(Integer)        # 연봉 인상률
    stock_option_level = Column(Integer)

    # 경력
    job_level = Column(Integer)                  # 직급 수준 (1~5)
    total_working_years = Column(Integer)
    num_companies_worked = Column(Integer)       # 근무한 회사 수
    years_at_company = Column(Integer)
    years_in_current_role = Column(Integer)
    years_since_last_promotion = Column(Integer) # 마지막 승진 이후 연수
    years_with_curr_manager = Column(Integer)    # 현재 관리자와 함께한 연수
    training_times_last_year = Column(Integer)   # 작년 교육 횟수

    # 직무
    job_role = Column(String)
    department = Column(String)

    # 근무 환경
    business_travel = Column(String)             # Travel_Rarely=가끔 출장, Travel_Frequently=자주 출장
    distance_from_home = Column(Integer)         # 통근 거리
    over_time = Column(Boolean)

    # 만족도
    environment_satisfaction = Column(Integer)   # 근무 환경 만족도
    job_satisfaction = Column(Integer)           # 직무 만족도
    relationship_satisfaction = Column(Integer)  # 인간관계 만족도
    job_involvement = Column(Integer)            # 업무 몰입도
    performance_rating = Column(Integer)         # 성과 평가
    work_life_balance = Column(Integer)

    # target
    attrition = Column(Boolean)