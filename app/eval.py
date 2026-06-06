import joblib
from preprocess import preprocess
from sklearn.metrics import accuracy_score


model = joblib.load("../models/catboost_best_model.pkl")
X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess()

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.4).astype(int)

from sklearn.metrics import classification_report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))