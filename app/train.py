from preprocess import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    )
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

X_train, X_test, y_train, y_test, scaler = preprocess()

print(f"훈련 세트{X_train.shape[0]}")
print(f"테스트 세트{X_test.shape[0]}")


lr = LogisticRegression(
    max_iter=10000,
    class_weight="balanced"
    )

lr.fit(X_train, y_train)
y_lr_pred = lr.predict(X_test)

print("=" * 60)
print("로지스틱 회귀")
print("=" * 60)

# 결정 트리 평가
print(f"정확도{accuracy_score(y_test, y_lr_pred):.4f}")
print(classification_report(y_test, y_lr_pred))

print(y_train.value_counts())
print(confusion_matrix(y_test, y_lr_pred))


# 단일 결정 트리
dt = DecisionTreeClassifier(
                            class_weight="balanced",
                            random_state=42
                            )

dt.fit(X_train, y_train)
y_dt_pred = dt.predict(X_test)

print("=" * 60)
print("결정 트리")
print("=" * 60)

# 결정 트리 평가
print(f"정확도{accuracy_score(y_test, y_dt_pred):.4f}")
print(classification_report(y_test, y_dt_pred))

print(y_train.value_counts())
print(confusion_matrix(y_test, y_dt_pred))


print("=" * 60)
print("랜덤 포레스트")
print("=" * 60)

# 랜덤 포레스트
rf = RandomForestClassifier(
    class_weight="balanced", # 클래스 불균형 처리
    random_state=42,
    )

rf.fit(X_train, y_train)
y_rf_pred = rf.predict(X_test)

print(f"정확도{accuracy_score(y_test, y_rf_pred):.4f}")
print(classification_report(y_test, y_rf_pred))

print(y_train.value_counts())   
print(confusion_matrix(y_test, y_rf_pred))

print("=" * 60)
print("AdaBoost")
print("=" * 60)


ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(class_weight="balanced"),
    random_state=42,
)

ada.fit(X_train, y_train)
y_ada_pred = ada.predict(X_test)

print(f"정확도{accuracy_score(y_test, y_ada_pred):.4f}")
print(classification_report(y_test, y_ada_pred))

print(y_train.value_counts())   
print(confusion_matrix(y_test, y_ada_pred))

print("=" * 60)
print("XGBoost")
print("=" * 60)

xgb = XGBClassifier(
    scale_pos_weight=978/198,
)

xgb.fit(X_train, y_train)
y_xgb_pred = xgb.predict(X_test)

print(f"정확도{accuracy_score(y_test, y_xgb_pred):.4f}")
print(classification_report(y_test, y_xgb_pred))

print(y_train.value_counts())   
print(confusion_matrix(y_test, y_xgb_pred))

# 각 모델을 비교 시 Logistic Regression은 Recall이 높아 퇴사자 예측은 유리했으나 False Positive가 많다.
# XGBoost는 precision과 recall의 균형이 우후하다.

# 추가로 svm도 실험
print("=" * 60)
print("SVM")
print("=" * 60)

svm = SVC(
    class_weight="balanced",
    random_state=42,
)

svm.fit(X_train, y_train)
y_svm_pred = svm.predict(X_test)

print(f"정확도{accuracy_score(y_test, y_svm_pred):.4f}")
print(classification_report(y_test, y_svm_pred))

print(y_train.value_counts())   
print(confusion_matrix(y_test, y_svm_pred))
