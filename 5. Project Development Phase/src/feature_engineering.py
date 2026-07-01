import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype

# -----------------------------------
# Load Dataset
# -----------------------------------
dataset_path = os.path.join("..", "Dataset", "Application_Data.csv")
df = pd.read_csv(dataset_path)

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

print("\nOriginal Dataset Shape :", df.shape)


# Remove leading/trailing spaces from all text columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.strip()


# -----------------------------------
# Remove Extra Spaces from String Columns
# -----------------------------------
for column in df.columns:
    if not is_numeric_dtype(df[column]):
        df[column] = df[column].astype(str).str.strip()

# -----------------------------------
# Remove Duplicate Rows
# -----------------------------------
duplicates = df.duplicated().sum()

if duplicates > 0:
    print(f"\nRemoving {duplicates} duplicate rows...")
    df.drop_duplicates(inplace=True)
else:
    print("\nNo duplicate rows found.")

print("Dataset Shape After Removing Duplicates :", df.shape)

# -----------------------------------
# Handle Missing Values
# -----------------------------------
for column in df.columns:
    if is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())
    else:
        df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing Values Handled Successfully.")

# -----------------------------------
# Encode Categorical Features
# -----------------------------------
label_encoders = {}

categorical_columns = [
    "Applicant_Gender",
    "Income_Type",
    "Education_Type",
    "Family_Status",
    "Housing_Type",
    "Job_Title"
]

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

print("\nCategorical Columns Encoded Successfully.")

# -----------------------------------
# Save Encoders
# -----------------------------------
os.makedirs("../model", exist_ok=True)
joblib.dump(label_encoders, "../model/label_encoders.pkl")

print("Label Encoders Saved.")

# -----------------------------------
# Save Processed Dataset
# -----------------------------------
processed_dataset = os.path.join(
    "..",
    "Dataset",
    "processed_application_data.csv"
)

df.to_csv(processed_dataset, index=False)

print("\nProcessed Dataset Saved Successfully.")
print("\nProcessed Dataset Shape :", df.shape)

print("=" * 60)
print("Feature Engineering Completed Successfully!")
print("=" * 60)