import joblib
import pandas as pd

# ---------------------------------------------------
# Load Saved Model, Scaler and Label Encoders
# ---------------------------------------------------

model = joblib.load("../model/best_model.pkl")
scaler = joblib.load("../model/scaler.pkl")
label_encoders = joblib.load("../model/label_encoders.pkl")

# Debug (optional)
print("Applicant_Gender classes:", label_encoders["Applicant_Gender"].classes_)

# ---------------------------------------------------
# Enter Applicant Details
# ---------------------------------------------------

sample = {
    "Applicant_ID": 5009999,
    "Applicant_Gender": "M",
    "Owned_Car": 1,
    "Owned_Realty": 1,
    "Total_Children": 1,
    "Total_Income": 180000,
    "Income_Type": "Working",
    "Education_Type": "Higher education",
    "Family_Status": "Married",
    "Housing_Type": "House / apartment",
    "Owned_Mobile_Phone": 1,
    "Owned_Work_Phone": 1,
    "Owned_Phone": 1,
    "Owned_Email": 1,
    "Job_Title": "Managers",
    "Total_Family_Members": 3,
    "Applicant_Age": 35,
    "Years_of_Working": 10,
    "Total_Bad_Debt": 0,
    "Total_Good_Debt": 20
}

# ---------------------------------------------------
# Convert to DataFrame
# ---------------------------------------------------

input_df = pd.DataFrame([sample])

# ---------------------------------------------------
# Remove Extra Spaces
# ---------------------------------------------------

categorical_columns = [
    "Applicant_Gender",
    "Income_Type",
    "Education_Type",
    "Family_Status",
    "Housing_Type",
    "Job_Title"
]

for column in categorical_columns:
    input_df[column] = input_df[column].astype(str).str.strip()

# ---------------------------------------------------
# Encode Categorical Columns
# ---------------------------------------------------

for column in categorical_columns:
    try:
        input_df[column] = label_encoders[column].transform(input_df[column])
    except ValueError:
        print(f"\nERROR: '{input_df[column].iloc[0]}' is not present in the encoder for '{column}'.")
        print("Available values:", list(label_encoders[column].classes_))
        exit()

# ---------------------------------------------------
# Scale Features
# ---------------------------------------------------

input_scaled = scaler.transform(input_df)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

prediction = model.predict(input_scaled)[0]

print("\n" + "=" * 60)

if prediction == 1:
    print("Credit Card Status : APPROVED")
else:
    print("Credit Card Status : REJECTED")

print("=" * 60)

# ---------------------------------------------------
# Prediction Probability
# ---------------------------------------------------

if hasattr(model, "predict_proba"):
    probability = model.predict_proba(input_scaled)
    print("\nPrediction Probability:")
    print(probability)