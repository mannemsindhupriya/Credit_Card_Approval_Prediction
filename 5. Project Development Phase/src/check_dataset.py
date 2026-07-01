"""
import pandas as pd

# Load the dataset
df = pd.read_csv("../Dataset/Application_Data.csv")

# Print all column names
print(df.columns.tolist())
"""

import pandas as pd

df = pd.read_csv("../Dataset/Application_Data.csv")

print(df["Applicant_Gender"].unique())
print(df["Income_Type"].unique())