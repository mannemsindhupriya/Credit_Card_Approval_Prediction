import joblib
from xgboost import XGBClassifier
from utils import evaluate_model, save_model

print("=" * 60)
print("XGBOOST MODEL")
print("=" * 60)

# Load Data
X_train = joblib.load("../model/X_train.pkl")
X_test = joblib.load("../model/X_test.pkl")
y_train = joblib.load("../model/y_train.pkl")
y_test = joblib.load("../model/y_test.pkl")

print("\nDataset Loaded Successfully")

# Train Model
model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# Evaluate Model
evaluate_model(model, X_test, y_test)

# Save Model
save_model(model, "xgboost_model")

print("\nXGBoost Model Completed Successfully")