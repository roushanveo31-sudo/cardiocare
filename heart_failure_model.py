
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_heart_failure_model():
    """
    Trains a heart failure prediction model using the heart_failure_clinical_records_dataset.csv dataset.
    """
    print("Training Heart Failure Model...")

    # Load the dataset
    try:
        df = pd.read_csv('archive (2)/heart_failure_clinical_records_dataset.csv')
    except FileNotFoundError:
        print("Error: heart_failure_clinical_records_dataset.csv not found. Make sure it's in the 'archive (2)/' directory.")
        return

    # Define features and target
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Heart Failure Model Accuracy: {accuracy:.4f}")

    # Save the model
    joblib.dump(model, 'heart_failure_model.joblib')
    print("Heart Failure Model saved as heart_failure_model.joblib")

if __name__ == '__main__':
    train_heart_failure_model()
