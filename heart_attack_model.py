
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

def train_heart_attack_model():
    """
    Trains a heart attack risk prediction model and saves both the model and the gender encoder.
    """
    print("Training Heart Attack Risk Model...")

    try:
        df = pd.read_csv('archive (1)/heart_attack_prediction_india.csv')
    except FileNotFoundError:
        print("Error: heart_attack_prediction_india.csv not found.")
        return

    df = df.drop(columns=['Patient_ID', 'State_Name'])

    # --- FIX: Save the LabelEncoder ---
    # We will save the encoder so the web app can use the exact same encoding.
    gender_encoder = LabelEncoder()
    df['Gender'] = gender_encoder.fit_transform(df['Gender'])

    X = df.drop(columns=['Heart_Attack_Risk'])
    y = df['Heart_Attack_Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Heart Attack Risk Model Accuracy: {accuracy:.4f}")

    # Save both the model and the encoder
    joblib.dump(model, 'heart_attack_model.joblib')
    joblib.dump(gender_encoder, 'gender_encoder.joblib')
    print("Heart Attack Risk Model and Gender Encoder saved.")

if __name__ == '__main__':
    train_heart_attack_model()
