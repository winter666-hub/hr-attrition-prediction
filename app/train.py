from preprocess import preprocess
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
    )
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV)
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

X_train, X_test, y_train, y_test, scaler = preprocess()

print(f"훈련 세트{X_train.shape[0]}")
print(f"테스트 세트{X_test.shape[0]}")

# evaluate 함수
def evaluate_model (name, y_test, y_pred):
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"정확도{accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# 모델 딕셔너리
models = {
    "Logistic Regression":    LogisticRegression(max_iter=10000, class_weight="balanced"),
    "DecisionTreeClassifier": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "RandomForestClassifier": RandomForestClassifier(class_weight="balanced", random_state=42),
    "AdaBoostClassifier":     AdaBoostClassifier(estimator=DecisionTreeClassifier(class_weight="balanced"), random_state=42),
    "XGBClassifier" :         XGBClassifier(random_state=42),
    "SVM" :                   SVC(class_weight="balanced", random_state=42),
    "KNN" :                   KNeighborsClassifier(),
    "GBM" :                   GradientBoostingClassifier(random_state=42),
    "Light GBM" :             LGBMClassifier(random_state=42),
}

# 퇴사자 수 부족으로 SMOTE 적용
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 반복문으로 학습
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    evaluate_model(name, y_test, y_pred)


# ===== 기본값 결과 =====
# 모델          정확도      Recall(True)        F1(True)
# Logistic       0.72         0.59               0.36
# RandomForest   0.88         0.10               0.19
# XGBoost        0.87         0.31               0.39
# SVM            0.85         0.46               0.44


# ===== 로지스틱 회귀 튜닝 =====

lr_params = {
    "C": [0.003, 0.01, 0.03],    # 규제 강도
    "solver": ["liblinear", "lbfgs"]
}

lr_grid = GridSearchCV(
    LogisticRegression(
        max_iter=10000,
        class_weight="balanced",
        ),
    lr_params,
    cv=5,
    scoring="recall",
    n_jobs=-1,
    verbose=1
)

lr_grid.fit(X_train_sm, y_train_sm)
print(lr_grid.best_params_)


best_lr = lr_grid.best_estimator_
y_lr_tuned_pred = best_lr.predict(X_test)
evaluate_model("Logistic Regression (tuned)", y_test, y_lr_tuned_pred)

y_lr_prob = best_lr.predict_proba(X_test)[:, 1]

# AUC 계산
lr_auc = roc_auc_score(y_test, y_lr_prob)

print("=" * 60)
print(f"Logistic Regression AUC: {lr_auc:.4f}")
print("=" * 60)

# ROC Curve
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_lr_prob)

plt.figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, label=f"Logistic Regression (AUC = {lr_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()

plt.savefig("../models/roc_lr.png")

for threshold in [0.5, 0.4, 0.3, 0.25]:
    y_lr_pred = (y_lr_prob >= threshold).astype(int)
    evaluate_model(f"Logistic threshold={threshold}",y_test, y_lr_pred)

rf_params = {
    "n_estimators": [100, 200],         # 트리 개수
    "max_depth": [5, 7, 9],             # 트리 최대 깊이
    "min_samples_split": [2, 5, 10],    # 노드 분할 최소 샘플 수
    "min_samples_leaf": [1, 3, 5]       # 리프 노드 최소 샘플 수
}
rf_grid = GridSearchCV(
    RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
        ),
    rf_params,
    cv=5,
    scoring="recall",
    n_jobs=-1
)

rf_grid.fit(X_train_sm, y_train_sm)
print(rf_grid.best_params_)

# Best Params: {'max_depth': 5, 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 100}

best_rf = rf_grid.best_estimator_
y_rf_tuned_pred = best_rf.predict(X_test)
evaluate_model("RandomForest (tuned)", y_test, y_rf_tuned_pred)

# 정확도 0.82, Recall(True): 0.31, F1(True): 0.31

# Threshold 조정

# 확률값 추출
y_rf_prob = best_rf.predict_proba(X_test)[:, 1]

for threshold in [0.5, 0.4, 0.3, 0.25]:
    y_rf_pred = (y_rf_prob >= threshold).astype(int)
    evaluate_model(f"RandomForest threshold={threshold}", y_test, y_rf_pred)


# ===== XGBoost 튜닝 =====
xgb_params = {
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "min_child_weight": [1, 3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma": [0, 0.1],
    "scale_pos_weight": [1, 2, 3]
}

xgb_grid = RandomizedSearchCV(
    XGBClassifier(
        random_state=42,
        eval_metric="logloss",
    ),
    xgb_params,
    n_iter=30,
    cv=5,
    scoring="recall",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train_sm, y_train_sm)
print(xgb_grid.best_params_)

# Best Params: {'learning_rate': 0.03, 'max_depth': 1, 'n_estimators': 300}

y_xgb_tuned_pred = xgb_grid.best_estimator_.predict(X_test)
evaluate_model("XGBoost (tuned)", y_test, y_xgb_tuned_pred)

# 정확도 0.75, Recall(True): 0.62, F1(True): 0.40

# threshold 조정

# 확률값 추출
y_xgb_prob = xgb_grid.best_estimator_.predict_proba(X_test)[:, 1]

for threshold in [0.5, 0.4, 0.3, 0.25]:
    y_xgb_pred = (y_xgb_prob >= threshold).astype(int)
    evaluate_model(f"XGBoost threshold={threshold}", y_test, y_xgb_pred)

# ===== SVM 튜닝 =====
svm_params = {
    "C": [1, 10, 100, 1000],      # 오분류 허용 정도
    "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1],
    "kernel": ["rbf"]
}

svm_grid = RandomizedSearchCV(
    SVC(
        class_weight="balanced",
        probability=True,
        random_state=42
    ),
    svm_params,
    n_iter=20,
    cv=3,
    scoring="recall",
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

# 학습
svm_grid.fit(X_train_sm, y_train_sm)
print(svm_grid.best_params_)

# Best Params :{'C': 0.1, 'gamma': 0.001, 'kernel': 'rbf'}

y_svm_tuned_pred = svm_grid.best_estimator_.predict(X_test)
evaluate_model("SVM (tuned)", y_test, y_svm_tuned_pred)

# 확률값 추출
y_svm_prob = svm_grid.best_estimator_.predict_proba(X_test)[:, 1]
# AUC 계산
svm_auc = roc_auc_score(y_test, y_svm_prob)

print("=" * 60)
print(f"SVM AUC: {svm_auc:.4f}")
print("=" * 60)

# ROC Curve
svm_fpr, svm_tpr, _ = roc_curve(y_test, y_svm_prob)

plt.figure(figsize=(8, 6))
plt.plot(svm_fpr, svm_tpr, label=f"SVM (AUC = {svm_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")
plt.legend()

plt.savefig("../models/roc_svm.png")

for threshold in [0.5, 0.4, 0.3, 0.25]:
    y_svm_pred = (y_svm_prob >= threshold).astype(int)
    evaluate_model(f"SVM threshold={threshold}", y_test, y_svm_pred)

# ===== 최종 결과 =====
# 모델                  AUC     Recall  F1
# LR (tuned)           0.7888   0.72    0.43
# SVM (tuned)          0.7752   0.64    0.42

cat_params = {
    "iterations": [150, 200, 250,],
    "learning_rate": [0.03, 0.05, 0.07, 0.1],
    "depth": [2, 3, 4],
    "l2_leaf_reg": [5, 7, 9, 11],
    "border_count": [32, 64, 128],
    "subsample": [0.6, 0.8, 1.0]
}

cat_grid = RandomizedSearchCV(
    CatBoostClassifier(
        loss_function="Logloss",
        random_state=42,
        verbose=0
    ),
    cat_params,
    n_iter=20,
    cv=5,
    scoring="recall",
    n_jobs=-1,
    random_state=42
)

cat_grid.fit(X_train_sm, y_train_sm)
print(cat_grid.best_params_)

y_cat_tuned_pred = cat_grid.best_estimator_.predict(X_test)
evaluate_model("CatBoost (tuned)", y_test, y_cat_tuned_pred)


y_cat_prob = cat_grid.best_estimator_.predict_proba(X_test)[:, 1]

for threshold in [0.5, 0.4, 0.3, 0.25]:
    y_cat_pred = (y_cat_prob >= threshold).astype(int)
    evaluate_model(f"CatBoost threshold={threshold}", y_test, y_cat_pred)
