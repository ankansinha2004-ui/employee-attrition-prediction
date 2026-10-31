import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the files saved at the end of the notebook
model = joblib.load("employee_attrition_model.pkl")
scaler = joblib.load("scaler.pkl")

# Same column order the model was trained on
FEATURES = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "Department",
    "salary",
]

# (min, max) allowed for each field
LIMITS = {
    "satisfaction_level": (0, 1),
    "last_evaluation": (0, 1),
    "number_project": (1, 10),
    "average_montly_hours": (50, 350),
    "time_spend_company": (1, 40),
    "Work_accident": (0, 1),
    "promotion_last_5years": (0, 1),
    "Department": (0, 9),
    "salary": (0, 2),
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    row = {}

    for name in FEATURES:
        if data.get(name) in (None, ""):
            return jsonify(error=f"Enter a value for {name}."), 400
        try:
            value = float(data[name])
        except (TypeError, ValueError):
            return jsonify(error=f"{name} must be a number."), 400
        low, high = LIMITS[name]
        if not low <= value <= high:
            return jsonify(error=f"{name} must be between {low} and {high}."), 400
        row[name] = value

    df = pd.DataFrame([row])[FEATURES]
    scaled = scaler.transform(df)
    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1])

    return jsonify(
        prediction=prediction,
        label="Likely to leave" if prediction == 1 else "Likely to stay",
        probability=round(probability * 100, 1),
    )


if __name__ == "__main__":
    app.run(debug=True)
