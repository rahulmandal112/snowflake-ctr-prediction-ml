import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


#  Load Dataset

df = pd.read_csv("combine5.csv")

print("Dataset Shape:", df.shape)
print("Click Rate:", df["clicked"].mean())


#  Feature / Target Split

X = df.drop("clicked", axis=1)
y = df["clicked"]

# One-hot encode categorical features
X = pd.get_dummies(
    X,
    columns=["user_location", "device_type", "ad_category", "time_of_day"]
)


#  Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#  Handle Class Imbalance

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos

print("Scale Pos Weight:", scale_pos_weight)


#  Train XGBoost
model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

model.fit(X_train, y_train)


#  Evaluate Model

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_prob)

print("\n=== XGBOOST PERFORMANCE ===")
print("AUC:", auc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# Feature Importance

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop 10 Important Features:\n")
print(importances.head(10))

plt.figure(figsize=(8, 6))
importances.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()