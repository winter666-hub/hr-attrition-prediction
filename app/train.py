from preprocess import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    classification_report,
    f1_score,
    precision_score
)

X_train, X_test, y_train, y_test, scaler = preprocess()

print(f"훈련 세트{X_train.shape[0]}")
print(f"테스트 세트{X_test.shape[0]}")

# 로지스틱 회귀 모델 생성
model = LogisticRegression(max_iter=10000,
                           class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 예측 평가
print(f"정확도{accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

print(y_train.value_counts())