from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and columns
model, trained_columns = joblib.load("student_performance_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # try:
        # Retrieve form data
        study_time = float(request.form['study_time'])
        failures = int(request.form['failures'])
        absences = int(request.form['absences'])
        famrel = int(request.form['famrel'])
        internet = 1 if request.form['internet'].lower() == "yes" else 0

        # Input for prediction
        input_features = pd.DataFrame([[
            study_time,
            failures,
            absences,
            famrel,
            internet
        ]], columns=['study_time', 'failures', 'absences', 'famrel', 'internet'])

        # Transform input features
        input_features_transformed = pd.get_dummies(input_features)
        input_features_transformed = input_features_transformed.reindex(columns=trained_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_features_transformed)[0]
        print(f"Prediction result: {prediction}")

        # Convert prediction to integer
        prediction = int(prediction)

        # Return JSON response
        result = {"predicted_grade": prediction, "status": "Success"}
        print(result)
        return jsonify(result)

    # except Exception as e:
        # Handle errors gracefully
        # return jsonify({"error": str(e), "status": "Failed"})

if __name__ == "__main__":
    app.run(debug=True)
