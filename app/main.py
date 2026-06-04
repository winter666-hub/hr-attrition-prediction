from fastapi import FastAPI
import joblib

app = FastAPI()

@app.get("/")
def root():
    return {"message": "HR Attrition Prediction API"}
