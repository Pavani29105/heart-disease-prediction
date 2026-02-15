from flask import Flask, render_template, request
import joblib
import pandas as pd
import sqlite3
from datetime import datetime

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect("heart.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            sex TEXT,
            cholesterol INTEGER,
            prediction TEXT,
            probability TEXT,
            date TEXT
        )
    """)

    conn.commit()
    conn.close()

model = joblib.load("heart_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Get form data
    name = request.form["name"]
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

    # Create dataframe for model
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

    # Prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    if prediction == 1:
        result = "⚠️ High Risk of Heart Disease"
        probability = f"{probabilities[1] * 100:.2f}% probability of Heart Disease"
    else:
        result = "✅ Low Risk of Heart Disease"
        probability = f"{probabilities[0] * 100:.2f}% probability of No Heart Disease"

    conn = sqlite3.connect("heart.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO patients (name, age, sex, cholesterol, prediction, probability, date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        age,
        sex,
        cholesterol,
        result,
        probability,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        result=result,
        probability=probability,
        name=name
    )

@app.route("/history")
def history():
    conn = sqlite3.connect("heart.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patients ORDER BY id ASC")
    records = cursor.fetchall()

    conn.close()

    return render_template("history.html", records=records)

if __name__ == "__main__":
    init_db()  # Create table if not exists
    app.run(debug=True)
