from flask import Flask, render_template, request
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------
try:
    model = joblib.load("../model/best_model.pkl")
    scaler = joblib.load("../model/scaler.pkl")
    label_encoders = joblib.load("../model/label_encoders.pkl")
except Exception as e:
    print("Error loading model files:", e)
    model = None

categorical_columns = [
    "Applicant_Gender",
    "Income_Type",
    "Education_Type",
    "Family_Status",
    "Housing_Type",
    "Job_Title"
]

# -----------------------------------------------------
# Home Page
# -----------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------------------------------
# Prediction
# -----------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return render_template(
            "result.html",
            prediction="Application Error: Model files could not be loaded."
        )

    try:

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

        # ------------------------
        # Input Validation
        # ------------------------

        if data["Applicant_Age"] <= 0:
            raise ValueError("Applicant Age must be greater than 0.")

        if data["Total_Income"] < 0:
            raise ValueError("Income cannot be negative.")

        if data["Years_of_Working"] < 0:
            raise ValueError("Years of working cannot be negative.")

        if data["Total_Children"] < 0:
            raise ValueError("Total children cannot be negative.")

        if data["Total_Family_Members"] <= 0:
            raise ValueError("Family members must be greater than zero.")

        if data["Total_Bad_Debt"] < 0 or data["Total_Good_Debt"] < 0:
            raise ValueError("Debt values cannot be negative.")

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Encode categorical columns
        for col in categorical_columns:
            df[col] = label_encoders[col].transform(df[col])

        # Scale data
        scaled_data = scaler.transform(df)

        # Predict
        prediction = model.predict(scaled_data)[0]

        if prediction == 1:
            result = "Credit Card Approved ✅"
        else:
            result = "Credit Card Rejected ❌"

        return render_template("result.html", prediction=result)

    # ------------------------
    # Invalid Values
    # ------------------------
    except ValueError as e:
        return render_template(
            "result.html",
            prediction=f"Input Error: {e}"
        )

    # ------------------------
    # Missing Form Field
    # ------------------------
    except KeyError as e:
        return render_template(
            "result.html",
            prediction=f"Missing Input Field: {e}"
        )

    # ------------------------
    # Unknown Category
    # ------------------------
    except Exception:
        print(traceback.format_exc())
        return render_template(
            "result.html",
            prediction="An unexpected error occurred. Please try again."
        )


# -----------------------------------------------------
# Custom 404 Error
# -----------------------------------------------------
@app.errorhandler(404)
def page_not_found(error):
    return render_template(
        "result.html",
        prediction="404 Error: Page Not Found."
    ), 404


# -----------------------------------------------------
# Custom 500 Error
# -----------------------------------------------------
@app.errorhandler(500)
def internal_server_error(error):
    return render_template(
        "result.html",
        prediction="500 Error: Internal Server Error."
    ), 500


# -----------------------------------------------------
# Run App
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)