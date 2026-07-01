import joblib

from sklearn.tree import DecisionTreeClassifier

from utils import evaluate_model, save_model

print("="*60)
print("DECISION TREE MODEL")
print("="*60)

# ----------------------------------------
# Load Data
# ----------------------------------------

X_train = joblib.load("../model/X_train.pkl")
X_test = joblib.load("../model/X_test.pkl")

y_train = joblib.load("../model/y_train.pkl")
y_test = joblib.load("../model/y_test.pkl")

print("\nDataset Loaded Successfully")

# ----------------------------------------
# Train Model
# ----------------------------------------

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10
)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ----------------------------------------
# Evaluation
# ----------------------------------------

evaluate_model(model, X_test, y_test)

# ----------------------------------------
# Save Model
# ----------------------------------------

save_model(model, "decision_tree_model")

print("\nDecision Tree Completed Successfully")