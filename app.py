from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("heart_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    sex = request.form["sex"]
    chest_pain = request.form["chest_pain"]
    resting_bp = int(request.form["resting_bp"])
    cholesterol = int(request.form["cholesterol"])
    fasting_bs = int(request.form["fasting_bs"])
    resting_ecg = request.form["resting_ecg"]
    max_hr = int(request.form["max_hr"])
    exercise_angina = request.form["exercise_angina"]
    oldpeak = float(request.form["oldpeak"])
    st_slope = request.form["st_slope"]

    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "⚠️ High Risk of Heart Disease"
    else:
        result = "✅ Low Risk of Heart Disease"

    # Send result to a new page
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
