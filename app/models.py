from sqlalchemy import Column, Integer, String
from database import Base

class Employees(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True)
    # 개인정보
    employee_number = Column(Integer)
    age = Column(Integer)
    gender = Column(String)