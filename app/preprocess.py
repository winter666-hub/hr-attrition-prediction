import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess():
    df = pd.read_csv("../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    # BusinessTravel
    travel_map = {
        "Non-Travel": 0,
        "Travel_Rarely": 1,
        "Travel_Frequently": 2
    }
    df["BusinessTravel"] = df["BusinessTravel"].map(travel_map)

    # Boolean
    df["OverTime"] = df["OverTime"] == "Yes"
    df["Attrition"] = df["Attrition"] == "Yes"

    # drop
    df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours"])

    # one-hot
    df = pd.get_dummies(
        df,
        columns=["Department", "EducationField", "Gender", "JobRole", "MaritalStatus"],
        drop_first=True
    )

    # split
    y = df["Attrition"]
    X = df.drop(columns=["Attrition"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler