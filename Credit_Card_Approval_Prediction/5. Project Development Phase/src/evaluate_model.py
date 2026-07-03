import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# --------------------------------------------------
# Load Test Data
# --------------------------------------------------

X_test = joblib.load("../model/X_test.pkl")
y_test = joblib.load("../model/y_test.pkl")

# --------------------------------------------------
# Load Models
# --------------------------------------------------

models = {
    "Logistic Regression": joblib.load("../model/logistic_model.pkl"),
    "Decision Tree": joblib.load("../model/decision_tree_model.pkl"),
    "Random Forest": joblib.load("../model/random_forest_model.pkl"),
    "XGBoost": joblib.load("../model/xgboost_model.pkl")
}

results = []

best_accuracy = 0
best_model = None
best_model_name = ""

# --------------------------------------------------
# Evaluate Every Model
# --------------------------------------------------

for name, model in models.items():

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# --------------------------------------------------
# Display Results
# --------------------------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\nModel Performance Comparison\n")
print(results_df)

# --------------------------------------------------
# Save Best Model
# --------------------------------------------------

joblib.dump(best_model, "../model/best_model.pkl")

print("\n" + "=" * 70)
print("Best Model :", best_model_name)
print("Accuracy   :", round(best_accuracy, 4))
print("=" * 70)

print("\nBest model saved as:")
print("../model/best_model.pkl")