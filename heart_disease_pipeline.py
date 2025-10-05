import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """
    Loads, cleans, preprocesses, and feature-engineers the four UCI heart disease datasets
    to create a single, powerful dataset for predicting coronary artery disease.
    """
    print("--- Starting Data Loading and Preprocessing ---")

    # 1. Load Data
    print("Loading 4 UCI datasets...")
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    df_cleveland = pd.read_csv('heart+disease/processed.cleveland.data', header=None, names=column_names)
    df_hungarian = pd.read_csv('heart+disease/processed.hungarian.data', header=None, names=column_names)
    df_switzerland = pd.read_csv('heart+disease/processed.switzerland.data', header=None, names=column_names)
    df_va = pd.read_csv('heart+disease/processed.va.data', header=None, names=column_names)

    df = pd.concat([df_cleveland, df_hungarian, df_switzerland, df_va], ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")

    # 2. Clean Data
    print("Cleaning data: handling '?' and converting types...")
    df.replace('?', np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Impute Missing Values
    print("Imputing missing values with column medians...")
    for col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

    # 4. Process Target Variable
    print("Processing target variable to binary format...")
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # 5. Feature Engineering (One-Hot Encoding)
    print("Performing one-hot encoding on categorical features...")
    df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

    print("--- Data Loading and Preprocessing Complete ---")
    return df

def main():
    """
    Main function to run the heart disease prediction pipeline.
    """
    print("===== Starting the World's Best Heart Disease Prediction Pipeline =====")

    # Load and preprocess the main dataset
    final_df = load_and_preprocess_data()

    print("\nFinal preprocessed DataFrame shape:", final_df.shape)
    print("Final preprocessed DataFrame head:")
    print(final_df.head())

    # --- Stacking Ensemble Model Implementation ---
    print("\n--- Defining the Stacking Ensemble Model ---")

    # 1. Define the base models
    base_estimators = [
        ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gradient_boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]

    # 2. Define the meta-model
    meta_model = LogisticRegression()

    # 3. Create the Stacking Classifier
    stacking_model = StackingClassifier(estimators=base_estimators, final_estimator=meta_model, cv=5)

    print("Stacking model defined successfully.")

    # --- Train and Evaluate the Final Model ---
    print("\n--- Training and Evaluating the Stacking Ensemble Model ---")

    X = final_df.drop('target', axis=1)
    y = final_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    stacking_model.fit(X_train_scaled, y_train)

    y_pred = stacking_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("\nFinal Stacking Model Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

    # --- Save the Final Model and Scaler ---
    print("\n--- Saving the final model and scaler ---")
    joblib.dump(stacking_model, 'best_heart_disease_model.joblib')
    joblib.dump(scaler, 'main_scaler.joblib')
    print("Model and scaler saved successfully as 'best_heart_disease_model.joblib' and 'main_scaler.joblib'.")

    print("\n===== Pipeline finished successfully. =====")

if __name__ == '__main__':
    main()