
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

    try:
        df = pd.read_csv('archive (2)/heart_failure_clinical_records_dataset.csv')
    except FileNotFoundError:
        print("Error: heart_failure_clinical_records_dataset.csv not found. Make sure it's in the 'archive (2)/' directory.")
        return

    # --- FIX: Remove the 'time' column ---
    # This feature is a follow-up duration, not a predictive clinical measure a user would have.
    # The model should not be trained on it for this application.
    X = df.drop(columns=['DEATH_EVENT', 'time'])
    y = df['DEATH_EVENT']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Heart Failure Model Accuracy: {accuracy:.4f}")

    joblib.dump(model, 'heart_failure_model.joblib')
    print("Heart Failure Model saved as heart_failure_model.joblib")

if __name__ == '__main__':
    train_heart_failure_model()
