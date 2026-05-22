# HR Attrition Prediction

직원의 인사 데이터를 기반으로 이직 여부를 예측하는 머신러닝 프로젝트입니다.

---

## Project Goal
IBM HR Analytics 데이터셋을 활용하여 직원의 이직 가능성을 예측하는 분류 모델을 구현합니다.  
단순 모델 학습에 그치지 않고, PostgreSQL DB 연동 및 데이터 파이프라인 구성까지 포함합니다.

---

## Tech Stack
- **Language**: Python
- **Data**: Pandas, NumPy
- **ML**: scikit-learn
- **Database**: PostgreSQL, SQLAlchemy, Supabase
- **Environment**: dotenv, venv

---

## Project Structure

```
hr-attrition-prediction/
├── app/
│   ├── base.py          # SQLAlchemy Base 설정
│   ├── database.py      # DB 연결 설정
│   ├── models.py        # ORM 모델 정의
│   ├── insert_data.py   # CSV → DB 적재
│   ├── eda.py           # 탐색적 데이터 분석
│   ├── preprocess.py    # 데이터 전처리
│   └── train.py         # 모델 학습 및 평가
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── notebooks/
│   └── eda.ipynb
├── requirements.txt
└── README.md
```

---

## Progress
- [x] 데이터 수집 (IBM HR Analytics Dataset)
- [x] PostgreSQL DB 설계 및 SQLAlchemy 연결
- [x] CSV → DB 적재 파이프라인 구성
- [x] EDA
- [X] 데이터 전처리 (인코딩, 스케일링, train/test split))
- [ ] 모델 학습 및 성능 비교 (Random Forest, XGBoost 등)
- [ ] API 연동

---

## Models
- Logistic Regression
- (추가 예정)

---

## Future Work
- 다양한 모델 비교 (Random Forest, XGBoost 등)
- FastAPI 기반 예측 API 구현
- 웹 서비스화
