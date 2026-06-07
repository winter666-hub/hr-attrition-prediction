import joblib
import matplotlib.pyplot as plt
import pandas as pd
from preprocess import preprocess
from sklearn.metrics import accuracy_score


model = joblib.load("../models/catboost_best_model.pkl")
X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess()

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.4).astype(int)

from sklearn.metrics import classification_report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

feature_importance = model.get_feature_importance()
feature_name = feature_cols

fi_df = pd.DataFrame({
    'feature': feature_name,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(fi_df)

plt.figure(figsize=(10, 8))
plt.barh(fi_df['feature'][:15], fi_df['importance'][:15])
plt.xlabel("Importance")
plt.title("Top 15 Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("../models/feature_importance.png")
plt.close()