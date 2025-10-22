
from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# --- Load Models and Preprocessors ---
cardio_model = joblib.load('cardio_model.joblib')
cardio_scaler = joblib.load('cardio_scaler.joblib')
heart_attack_model = tf.keras.models.load_model('heart_attack_model.keras', compile=False)
heart_attack_preprocessor = joblib.load('heart_attack_preprocessor.joblib')
heart_disease_model = joblib.load('heart_disease_model.joblib')
heart_disease_scaler = joblib.load('heart_disease_scaler.joblib')

# --- Helper Function for Detailed Report ---
def generate_report(risk_percentage, model_name):
    """Generates a detailed prediction report."""
    report = {
        "risk_percentage": f"{risk_percentage:.2f}%",
        "disclaimer": "This is a prediction based on a machine learning model and is not a substitute for professional medical advice. Please consult a doctor for any health concerns.",
        "suggestions": [
            "Maintain a balanced diet rich in fruits, vegetables, and whole grains.",
            "Engage in regular physical activity, aiming for at least 30 minutes most days of the week.",
            "Avoid smoking and limit alcohol consumption.",
            "Monitor your blood pressure and cholesterol levels regularly."
        ]
    }
    if model_name == "cardio":
        report["title"] = "Cardiovascular Disease Risk Report"
        if risk_percentage > 50:
            report["interpretation"] = "The model indicates a higher than average risk of cardiovascular disease. It is strongly recommended to consult a healthcare professional for a comprehensive evaluation."
        else:
            report["interpretation"] = "The model indicates a lower than average risk of cardiovascular disease. Continue to maintain a healthy lifestyle."
    elif model_name == "heart_attack":
        report["title"] = "Heart Attack Risk Report"
        if risk_percentage > 50:
            report["interpretation"] = "The model indicates a significant risk of a heart attack. Immediate consultation with a doctor is advised."
        else:
            report["interpretation"] = "The model indicates a lower risk of a heart attack. It is still important to follow a heart-healthy lifestyle."
    elif model_name == "heart_disease":
        report["title"] = "Heart Disease Diagnosis Report"
        if risk_percentage > 50:
            report["interpretation"] = "The model suggests a high probability of the presence of heart disease. Please see a medical professional for a formal diagnosis."
        else:
            report["interpretation"] = "The model suggests a low probability of the presence of heart disease. Continue to monitor your health with regular check-ups."
    return report

# --- Web Routes ---
@app.route('/')
def home():
    return render_template('index.html', prediction_report=None)

@app.route('/predict_cardio', methods=['POST'])
def predict_cardio():
    try:
        form_data = request.form.to_dict()
        features = [float(form_data[key]) for key in ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]
        input_data = pd.DataFrame([features], columns=['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
        input_scaled = cardio_scaler.transform(input_data)
        prediction_proba = cardio_model.predict_proba(input_scaled)[0][1]
        risk_percentage = prediction_proba * 100
        report = generate_report(risk_percentage, "cardio")
        return render_template('index.html', prediction_report=report, active_tab='cardio')
    except Exception as e:
        return render_template('index.html', prediction_report={"error": f"Error: {e}"}, active_tab='cardio')

@app.route('/predict_heart_attack', methods=['POST'])
def predict_heart_attack():
    try:
        form_data = request.form.to_dict()
        feature_dict = {
            'Age': int(form_data['Age']), 'Gender': form_data['Gender'], 'Diabetes': int(form_data['Diabetes']),
            'Hypertension': int(form_data['Hypertension']), 'Obesity': int(form_data['Obesity']), 'Smoking': int(form_data['Smoking']),
            'Alcohol_Consumption': int(form_data['Alcohol_Consumption']), 'Physical_Activity': int(form_data['Physical_Activity']),
            'Diet_Score': int(form_data['Diet_Score']), 'Cholesterol_Level': int(form_data['Cholesterol_Level']),
            'Triglyceride_Level': int(form_data['Triglyceride_Level']), 'LDL_Level': int(form_data['LDL_Level']),
            'HDL_Level': int(form_data['HDL_Level']), 'Systolic_BP': int(form_data['Systolic_BP']),
            'Diastolic_BP': int(form_data['Diastolic_BP']), 'Air_Pollution_Exposure': int(form_data['Air_Pollution_Exposure']),
            'Family_History': int(form_data['Family_History']), 'Stress_Level': int(form_data['Stress_Level']),
            'Healthcare_Access': int(form_data['Healthcare_Access']), 'Heart_Attack_History': int(form_data['Heart_Attack_History']),
            'Emergency_Response_Time': int(form_data['Emergency_Response_Time']), 'Annual_Income': int(form_data['Annual_Income']),
            'Health_Insurance': int(form_data['Health_Insurance'])
        }
        input_data = pd.DataFrame([feature_dict])
        input_processed = heart_attack_preprocessor.transform(input_data)
        prediction_prob = heart_attack_model.predict(input_processed)[0][0]
        risk_percentage = prediction_prob * 100
        report = generate_report(risk_percentage, "heart_attack")
        return render_template('index.html', prediction_report=report, active_tab='heart-attack')
    except Exception as e:
        return render_template('index.html', prediction_report={"error": f"Error: {e}"}, active_tab='heart-attack')

@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease():
    try:
        form_data = request.form.to_dict()
        features = [float(form_data[key]) for key in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
        input_data = pd.DataFrame([features], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        input_scaled = heart_disease_scaler.transform(input_data)
        prediction_proba = heart_disease_model.predict_proba(input_scaled)[0][1]
        risk_percentage = prediction_proba * 100
        report = generate_report(risk_percentage, "heart_disease")
        return render_template('index.html', prediction_report=report, active_tab='heart-disease')
    except Exception as e:
        return render_template('index.html', prediction_report={"error": f"Error: {e}"}, active_tab='heart-disease')

if __name__ == '__main__':
    app.run(debug=True)
