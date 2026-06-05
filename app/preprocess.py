import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 학습용 preprocess
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

    feature_cols = X_train.columns

    # scaling
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )


    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def transform_input(data, scaler, feature_cols):

    df = pd.DataFrame([data])

    # encoding
    travel_map = {
        "Non-Travel": 0,
        "Travel_Rarely": 1,
        "Travel_Frequently": 2
    }
    df["BusinessTravel"] = df["BusinessTravel"].map(travel_map)

    # Boolean
    df["OverTime"] = df["OverTime"] == "Yes"

    # one-hot encoding
    df = pd.get_dummies(
        df,
        columns=["Department", "EducationField", "Gender", "JobRole", "MaritalStatus"],
        drop_first=True
    )

    # 컬럼 맞추기
    df = df.reindex(columns=feature_cols, fill_value=0)

    # scaling
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_cols,
        index=df.index
    )

    return df_scaled