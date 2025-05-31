import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, ConfusionMatrixDisplay
)

data = pd.read_csv("preprocessed_dataset.csv")

'''
    Part 3. 주간 불안 점수를 기준으로 정신 건강 상태를 safe/Risk로 이진 분류하는 Classification
    Logistic Regression, Decision Tree
'''

# 2. 피처 & 타겟 정의
features = ['total_screen_time', 'sleep_efficiency', 'mental_health_index', 'wellness_score']
X = data[features]
threshold = data['weekly_anxiety_score'].median() # median 값을 기준으로 구분
y = (data['weekly_anxiety_score'] > threshold).astype(int)  # 1: 위험군 - risk, 0: 비위험군 - safe

# train/test 데이터 분할 및 standardzation ; Logistic을 위한 작업
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression - model train + predict 
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)
y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

# Decision tree - model train + predict
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

# 성능 평가 - Precision, Recall, F1-Score 등
print("< Logistic Regression >")
print(classification_report(y_test, y_pred_log))
print("< Decision Tree >")
print(classification_report(y_test, y_pred_tree))


# Confusion Matrix Visualization
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log, ax=ax[0], cmap='Blues')
ax[0].set_title("Logistic Regression - Confusion Matrix")

# Decision Tree Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tree, ax=ax[1], cmap='Greens')
ax[1].set_title("Decision Tree - Confusion Matrix")

plt.tight_layout()
plt.show()


# ROC Curve - model의 classification 성능 곡선 ; visualization -----------------> github 코드 참고
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
roc_auc_log = auc(fpr_log, tpr_log) # AUC 값 계산
roc_auc_tree = auc(fpr_tree, tpr_tree)

plt.figure(figsize=(7, 5))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {roc_auc_log:.2f})")
plt.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC = {roc_auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Decision Tree - Visualization
plt.figure(figsize=(24, 12))
plot_tree(
    tree_model,
    feature_names=features,
    class_names=["Safe", "Risk"],
    filled=True,
    rounded=True,
    fontsize=12,
    precision=2
)
plt.title("Decision Tree Structure (Readable)")
plt.tight_layout()
plt.savefig("decision_tree.pdf", bbox_inches='tight') # 다 담기지 않아서 pdf 추가로 저장하는 방식을 추가해 봄
plt.show()