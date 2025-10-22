
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# --- Load Models and Preprocessors ---

# Model 1: Cardiovascular Disease
cardio_model = joblib.load('cardio_model.joblib')
cardio_scaler = joblib.load('cardio_scaler.joblib')

# Model 2: Heart Attack Risk
heart_attack_model = tf.keras.models.load_model('heart_attack_model.keras', compile=False)
heart_attack_preprocessor = joblib.load('heart_attack_preprocessor.joblib')

# Model 3: Heart Disease Diagnosis
heart_disease_model = joblib.load('heart_disease_model.joblib')
heart_disease_scaler = joblib.load('heart_disease_scaler.joblib')

# --- Web Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_cardio', methods=['POST'])
def predict_cardio():
    try:
        # Get data from form
        form_data = request.form.to_dict()
        features = [float(form_data[key]) for key in ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]

        # Create a DataFrame and scale
        input_data = pd.DataFrame([features], columns=['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
        input_scaled = cardio_scaler.transform(input_data)

        # Predict
        prediction = cardio_model.predict(input_scaled)[0]
        result_text = 'High Risk of Cardiovascular Disease' if prediction == 1 else 'Low Risk of Cardiovascular Disease'

        return render_template('index.html', prediction_cardio=result_text)
    except Exception as e:
        return render_template('index.html', prediction_cardio=f'Error: {e}')

@app.route('/predict_heart_attack', methods=['POST'])
def predict_heart_attack():
    try:
        # Get data from form
        form_data = request.form.to_dict()

        # Prepare the DataFrame with correct types and column order
        feature_dict = {
            'Age': int(form_data['Age']),
            'Gender': form_data['Gender'],
            'Diabetes': int(form_data['Diabetes']),
            'Hypertension': int(form_data['Hypertension']),
            'Obesity': int(form_data['Obesity']),
            'Smoking': int(form_data['Smoking']),
            'Alcohol_Consumption': int(form_data['Alcohol_Consumption']),
            'Physical_Activity': int(form_data['Physical_Activity']),
            'Diet_Score': int(form_data['Diet_Score']),
            'Cholesterol_Level': int(form_data['Cholesterol_Level']),
            'Triglyceride_Level': int(form_data['Triglyceride_Level']),
            'LDL_Level': int(form_data['LDL_Level']),
            'HDL_Level': int(form_data['HDL_Level']),
            'Systolic_BP': int(form_data['Systolic_BP']),
            'Diastolic_BP': int(form_data['Diastolic_BP']),
            'Air_Pollution_Exposure': int(form_data['Air_Pollution_Exposure']),
            'Family_History': int(form_data['Family_History']),
            'Stress_Level': int(form_data['Stress_Level']),
            'Healthcare_Access': int(form_data['Healthcare_Access']),
            'Heart_Attack_History': int(form_data['Heart_Attack_History']),
            'Emergency_Response_Time': int(form_data['Emergency_Response_Time']),
            'Annual_Income': int(form_data['Annual_Income']),
            'Health_Insurance': int(form_data['Health_Insurance'])
        }
        input_data = pd.DataFrame([feature_dict])

        # Preprocess the data
        input_processed = heart_attack_preprocessor.transform(input_data)

        # Predict
        prediction_prob = heart_attack_model.predict(input_processed)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        result_text = 'High Risk of Heart Attack' if prediction == 1 else 'Low Risk of Heart Attack'

        return render_template('index.html', prediction_heart_attack=result_text)
    except Exception as e:
        return render_template('index.html', prediction_heart_attack=f'Error: {e}')

@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease():
    try:
        # Get data from form
        form_data = request.form.to_dict()
        features = [float(form_data[key]) for key in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]

        # Create a DataFrame and scale
        input_data = pd.DataFrame([features], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        input_scaled = heart_disease_scaler.transform(input_data)

        # Predict
        prediction = heart_disease_model.predict(input_scaled)[0]
        result_text = 'Heart Disease Presence Likely' if prediction == 1 else 'Heart Disease Presence Unlikely'

        return render_template('index.html', prediction_heart_disease=result_text)
    except Exception as e:
        return render_template('index.html', prediction_heart_disease=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
