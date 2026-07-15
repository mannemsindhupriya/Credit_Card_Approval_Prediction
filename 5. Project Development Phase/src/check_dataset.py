"""
import pandas as pd

# Load the dataset
df = pd.read_csv("../Dataset/Application_Data.csv")

# Print all column names
print(df.columns.tolist())
"""

import pandas as pd

# Load dataset
df = pd.read_csv("../Dataset/Application_Data.csv")

# Display first 5 rows
print(df.head())