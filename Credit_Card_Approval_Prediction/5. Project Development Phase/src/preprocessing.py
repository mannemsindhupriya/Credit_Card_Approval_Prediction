import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------
dataset_path = os.path.join("..", "Dataset", "Application_Data.csv")
df = pd.read_csv(dataset_path)

print("Dataset Loaded Successfully!")
print(df.head())

# Remove leading/trailing spaces from all text columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.strip()

# -----------------------------
# Check Missing Values
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())


print("\nColumn Data Types:")
print(df.dtypes)

# -----------------------------
# Fill Missing Values
# -----------------------------
from pandas.api.types import is_numeric_dtype

for column in df.columns:
    if is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())
    else:
        df[column] = df[column].fillna(df[column].mode()[0])
# -----------------------------
# Encode Categorical Columns
# -----------------------------
label_encoder = LabelEncoder()

for column in df.select_dtypes(include=["object", "str"]).columns:
    if column != "Status":
        df[column] = label_encoder.fit_transform(df[column])

# Encode Target Column
if not is_numeric_dtype(df["Status"]):
    df["Status"] = label_encoder.fit_transform(df["Status"])

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("Status", axis=1)
y = df["Status"]

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Save Scaler
# -----------------------------
os.makedirs("../model", exist_ok=True)
joblib.dump(scaler, "../model/scaler.pkl")

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", X_train.shape)
print("Testing Samples :", X_test.shape)

print("\nPreprocessing Completed Successfully!")


# -----------------------------
# Save Processed Data
# -----------------------------
joblib.dump(X_train, "../model/X_train.pkl")
joblib.dump(X_test, "../model/X_test.pkl")
joblib.dump(y_train, "../model/y_train.pkl")
joblib.dump(y_test, "../model/y_test.pkl")

print("\nProcessed data saved successfully!")