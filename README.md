# ❤️ Heart Disease Prediction System

A Machine Learning-based web application that predicts the risk of heart disease using patient health information. The application is built using Flask, Scikit-learn, SQLite, HTML, and CSS and is deployed on Render.

## 🚀 Features

- Predicts Heart Disease Risk using a trained Random Forest model
- Displays prediction result instantly
- Shows probability/confidence of prediction
- Stores patient records in SQLite database
- Maintains prediction history
- Simple and user-friendly interface
- Deployed online using Render

---

## 🛠️ Technologies Used

### Frontend
- HTML5
- CSS3

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- Random Forest Classifier
- Joblib

### Database
- SQLite

### Deployment
- GitHub
- Render

---

## 📊 Dataset Features

The model uses the following patient information:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak (ST Depression)
- ST Slope

---

## 📁 Project Structure

```text
heart-disease-prediction/
│
├── app.py
├── heart_model.pkl
├── heart.db
├── requirements.txt
├── Procfile
├── README.md
│
└── templates/
    ├── index.html
    ├── result.html
    └── history.html
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Pavani29105/heart-disease-prediction.git
cd heart-disease-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🎯 Working

1. User enters patient details.
2. Flask receives the input.
3. Trained Random Forest model processes the data.
4. Model predicts:
   - High Risk of Heart Disease
   - Low Risk of Heart Disease
5. Result is displayed to the user.
6. Patient information and prediction are stored in SQLite database.
7. Users can view previous records through the History page.

---

## 📈 Machine Learning Model

Algorithm Used:

- Random Forest Classifier

Advantages:

- High accuracy
- Handles categorical and numerical features
- Reduces overfitting
- Suitable for medical prediction systems

---

## 🗄️ Database

The application stores:

- Patient Name
- Prediction Result
- Probability
- Date & Time of Prediction

Database Used:

```text
SQLite (heart.db)
```

---

## 🌐 Deployment

The project is deployed using Render.

Live Application:

(Add your Render URL here)

Example:

```text
https://heart-disease-prediction-6kih.onrender.com
```

---

## 📷 Screenshots

Add screenshots of:

- Home Page
- <img width="2273" height="1442" alt="image" src="https://github.com/user-attachments/assets/f637a643-e3a0-4b92-b076-005ce44a1cc5" />

- Prediction Result Page
- <img width="778" height="640" alt="image" src="https://github.com/user-attachments/assets/016378f0-208b-406e-9d5b-55d046cb2a80" />

- 
- History Page
- <img width="2453" height="633" alt="image" src="https://github.com/user-attachments/assets/cdbac5b2-d57b-4108-a71e-735684fa9478" />


---

## 👩‍💻 Author

**Pavani Karnapu**

B.Tech Student

Machine Learning Mini Project

---

## 📜 License

This project is developed for educational and academic purposes.
