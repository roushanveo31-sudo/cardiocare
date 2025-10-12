from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path
from model_loader import load_all_models, MODEL_DIR

# --- App Initialization ---
app = Flask(__name__)

# --- Model Loading ---
load_all_models()

models = {}
try:
    models['heart_disease'] = joblib.load(MODEL_DIR / 'heart_disease_model.joblib')
    models['heart_attack'] = joblib.load(MODEL_DIR / 'heart_attack_model.joblib')
    models['heart_failure'] = joblib.load(MODEL_DIR / 'heart_failure_model.joblib')
    # --- FIX: Load the saved gender encoder ---
    models['gender_encoder'] = joblib.load(MODEL_DIR / 'gender_encoder.joblib')
    print("All models and encoders loaded successfully from local files.")
except FileNotFoundError as e:
    print(f"Error loading models: {e}. Make sure model/encoder files exist in the '{MODEL_DIR}' directory.")
    models = None

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if not models:
        return render_template('index.html', result="Error: Models are not loaded.")

    try:
        form_data = request.form.to_dict()
        form_data = {k: float(v) for k, v in form_data.items()}

        columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        input_df = pd.DataFrame([form_data], columns=columns)

        prediction = models['heart_disease'].predict(input_df)[0]
        result_text = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

    except Exception as e:
        result_text = f"An error occurred: {e}"

    return render_template('index.html', result=result_text)

@app.route('/predict_failure', methods=['POST'])
def predict_failure():
    if not models:
        return render_template('index.html', result="Error: Models are not loaded.")

    try:
        form_data = request.form.to_dict()
        form_data = {k: float(v) for k, v in form_data.items()}

        # --- FIX: Columns updated to remove 'time' ---
        columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                   'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']
        input_df = pd.DataFrame([form_data], columns=columns)

        prediction = models['heart_failure'].predict(input_df)[0]
        result_text = "High Risk of a Heart Failure Event" if prediction == 1 else "Low Risk of a Heart Failure Event"

    except Exception as e:
        result_text = f"An error occurred: {e}"

    return render_template('index.html', result=result_text)

@app.route('/predict_attack', methods=['POST'])
def predict_attack():
    if not models:
        return render_template('index.html', result="Error: Models are not loaded.")

    try:
        form_data = request.form.to_dict()

        gender_string = form_data.pop('Gender')
        form_data_numeric = {k: float(v) for k, v in form_data.items()}
        input_df = pd.DataFrame([form_data_numeric])

        # --- FIX: Use the loaded encoder for robust transformation ---
        # The input to transform must be a list or array-like
        gender_encoded = models['gender_encoder'].transform([gender_string])[0]
        input_df['Gender'] = gender_encoded

        columns = [
            'Age', 'Diabetes', 'Hypertension', 'Obesity', 'Smoking', 'Alcohol_Consumption',
            'Physical_Activity', 'Diet_Score', 'Cholesterol_Level', 'Triglyceride_Level',
            'LDL_Level', 'HDL_Level', 'Systolic_BP', 'Diastolic_BP', 'Air_Pollution_Exposure',
            'Family_History', 'Stress_Level', 'Healthcare_Access', 'Heart_Attack_History',
            'Emergency_Response_Time', 'Annual_Income', 'Health_Insurance', 'Gender'
        ]
        input_df = input_df[columns]

        prediction = models['heart_attack'].predict(input_df)[0]
        result_text = "High Risk of Heart Attack" if prediction == 1 else "Low Risk of Heart Attack"

    except Exception as e:
        result_text = f"An error occurred: {e}"

    return render_template('index.html', result=result_text)

if __name__ == '__main__':
    app.run(debug=True)
