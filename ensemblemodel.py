import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Load Data

df = pd.read_csv("combine5.csv")

print("Dataset Shape:", df.shape)
print("Click Rate:", df["clicked"].mean())


#  Feature / Target Split

X = df.drop("clicked", axis=1)
y = df["clicked"]

X = pd.get_dummies(
    X,
    columns=["user_location", "device_type", "ad_category", "time_of_day"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#  Define Models

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        random_state=42
    )
}


#  Train & Evaluate

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    results[name] = auc
    print(f"{name} AUC: {auc:.4f}")

print("\n=== MODEL COMPARISON ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")