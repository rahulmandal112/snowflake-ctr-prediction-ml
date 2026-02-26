import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt


# Load Dataset (Generated via Mockaroo)

df = pd.read_csv("combine5.csv")

print("Dataset Shape:", df.shape)
print("Click Rate:", df["clicked"].mean())


# Feature / Target Split

X = df.drop("clicked", axis=1)
y = df["clicked"]


# One-Hot Encode Categorical Columns

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
    stratify=y  # important for imbalanced data
)


# Train Logistic Regression Model

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
)

model.fit(X_train, y_train)


# Evaluate Model

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_prob)

print("\n=== MODEL PERFORMANCE ===")
print("AUC:", auc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


#  Feature Importance

importance = pd.Series(model.coef_[0], index=X.columns)
importance = importance.sort_values(ascending=False)

print("\nTop 10 Important Features:\n")
print(importance.head(10))

# Plot top features
plt.figure(figsize=(8, 6))
importance.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importance")
plt.tight_layout()
plt.show()