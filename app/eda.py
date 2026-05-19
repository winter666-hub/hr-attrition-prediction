import pandas as pd

df = pd.read_csv("../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(df.info())

print(df.select_dtypes(include="object").columns.tolist())

for col in ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus"]:
    print(f"{col}: {df[col].unique()}")

print(df["Gender"].unique())