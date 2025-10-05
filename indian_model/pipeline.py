import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    Loads and preprocesses the Indian heart disease dataset.
    """
    print("--- Starting Data Loading and Preprocessing for Indian Dataset ---")

    df = pd.read_csv('archive (1)/heart_attack_prediction_india.csv')

    # Drop non-predictive columns using their original names from the CSV
    df.drop(['Patient_ID', 'State_Name'], axis=1, inplace=True)

    # Standardize column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]

    # Convert 'gender' to numeric
    df['gender'] = df['gender'].apply(lambda x: 1 if x.lower() == 'male' else 0)

    # Rename target column
    df.rename(columns={'heart_attack_risk': 'target'}, inplace=True)

    print("--- Indian Data Loading and Preprocessing Complete ---")
    return df

def main():
    """
    Main function to run the Indian heart disease prediction pipeline.
    """
    print("===== Starting the Indian Heart Disease Prediction Pipeline =====")

    # Load and preprocess the dataset
    df_indian = load_and_preprocess_data()

    print("\nFinal preprocessed DataFrame shape:", df_indian.shape)
    print("Final preprocessed DataFrame head:")
    print(df_indian.head())

    # --- Train and Evaluate the Model ---
    print("\n--- Training and Evaluating the RandomForest Model ---")

    X = df_indian.drop('target', axis=1)
    y = df_indian['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("\nFinal Model Performance on Indian Dataset:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

    # --- Save the Final Model and Scaler ---
    print("\n--- Saving the final model and scaler for the Indian dataset ---")
    joblib.dump(model, 'indian_model/indian_model.joblib')
    joblib.dump(scaler, 'indian_model/indian_scaler.joblib')
    print("Model and scaler saved successfully to the 'indian_model/' directory.")

    print("\n===== Pipeline finished successfully. =====")

if __name__ == '__main__':
    main()