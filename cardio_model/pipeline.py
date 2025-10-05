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
    Loads and preprocesses the Cardio dataset.
    """
    print("--- Starting Data Loading and Preprocessing for Cardio Dataset ---")

    df = pd.read_csv('archive/cardio_train.csv', sep=';')

    # Drop the unnecessary id column
    df.drop('id', axis=1, inplace=True)

    # Rename the target column for consistency
    df.rename(columns={'cardio': 'target'}, inplace=True)

    print("--- Cardio Data Loading and Preprocessing Complete ---")
    return df

def main():
    """
    Main function to run the cardio disease prediction pipeline.
    """
    print("===== Starting the Cardio Disease Prediction Pipeline =====")

    # Load and preprocess the dataset
    df_cardio = load_and_preprocess_data()

    print("\nFinal preprocessed DataFrame shape:", df_cardio.shape)
    print("Final preprocessed DataFrame head:")
    print(df_cardio.head())

    # --- Train and Evaluate the Model ---
    print("\n--- Training and Evaluating the RandomForest Model ---")

    X = df_cardio.drop('target', axis=1)
    y = df_cardio['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("\nFinal Model Performance on Cardio Dataset:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

    # --- Save the Final Model and Scaler ---
    print("\n--- Saving the final model and scaler for the Cardio dataset ---")
    joblib.dump(model, 'cardio_model/cardio_model.joblib')
    joblib.dump(scaler, 'cardio_model/cardio_scaler.joblib')
    print("Model and scaler saved successfully to the 'cardio_model/' directory.")

    print("\n===== Pipeline finished successfully. =====")

if __name__ == '__main__':
    main()