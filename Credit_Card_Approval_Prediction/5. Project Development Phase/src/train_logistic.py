import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("LOGISTIC REGRESSION MODEL")
print("=" * 60)

# ----------------------------------------
# Load Training and Testing Data
# ----------------------------------------

X_train = joblib.load("../model/X_train.pkl")
X_test = joblib.load("../model/X_test.pkl")
y_train = joblib.load("../model/y_train.pkl")
y_test = joblib.load("../model/y_test.pkl")

print("\nData Loaded Successfully")

# ----------------------------------------
# Train Model
# ----------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ----------------------------------------
# Prediction
# ----------------------------------------

y_pred = model.predict(X_test)

# ----------------------------------------
# Evaluation
# ----------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nModel Performance")
print("-" * 30)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ----------------------------------------
# Save Model
# ----------------------------------------

os.makedirs("../model", exist_ok=True)

joblib.dump(model, "../model/logistic_model.pkl")

print("\nModel Saved Successfully!")

print("=" * 60)
print("Logistic Regression Completed Successfully!")
print("=" * 60)