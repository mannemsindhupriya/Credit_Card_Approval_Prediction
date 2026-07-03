import pandas as pd
import os

dataset_path = os.path.join("..", "Dataset", "Application_Data.csv")

df = pd.read_csv(dataset_path)

categorical_columns = [
    "Applicant_Gender",
    "Income_Type",
    "Education_Type",
    "Family_Status",
    "Housing_Type",
    "Job_Title"
]

for col in categorical_columns:
    print("\n" + "="*60)
    print(col)
    print("="*60)
    print(sorted(df[col].unique()))