from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------

model = joblib.load("../model/best_model.pkl")
scaler = joblib.load("../model/scaler.pkl")
label_encoders = joblib.load("../model/label_encoders.pkl")


categorical_columns = [
    "Applicant_Gender",
    "Income_Type",
    "Education_Type",
    "Family_Status",
    "Housing_Type",
    "Job_Title"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Applicant_ID": int(request.form["Applicant_ID"]),
        "Applicant_Gender": request.form["Applicant_Gender"],
        "Owned_Car": int(request.form["Owned_Car"]),
        "Owned_Realty": int(request.form["Owned_Realty"]),
        "Total_Children": int(request.form["Total_Children"]),
        "Total_Income": float(request.form["Total_Income"]),
        "Income_Type": request.form["Income_Type"],
        "Education_Type": request.form["Education_Type"],
        "Family_Status": request.form["Family_Status"],
        "Housing_Type": request.form["Housing_Type"],
        "Owned_Mobile_Phone": int(request.form["Owned_Mobile_Phone"]),
        "Owned_Work_Phone": int(request.form["Owned_Work_Phone"]),
        "Owned_Phone": int(request.form["Owned_Phone"]),
        "Owned_Email": int(request.form["Owned_Email"]),
        "Job_Title": request.form["Job_Title"],
        "Total_Family_Members": float(request.form["Total_Family_Members"]),
        "Applicant_Age": int(request.form["Applicant_Age"]),
        "Years_of_Working": int(request.form["Years_of_Working"]),
        "Total_Bad_Debt": int(request.form["Total_Bad_Debt"]),
        "Total_Good_Debt": int(request.form["Total_Good_Debt"])
    }

    df = pd.DataFrame([data])

    for col in categorical_columns:
        df[col] = label_encoders[col].transform(df[col])

    scaled_data = scaler.transform(df)

    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        result = "Credit Card Approved ✅"
    else:
        result = "Credit Card Rejected ❌"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)