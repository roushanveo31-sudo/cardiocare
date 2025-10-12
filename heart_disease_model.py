
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_heart_disease_model():
    """
    Trains a heart disease prediction model using the cardio_train.csv dataset.
    """
    print("Training Heart Disease Model...")

    # Load the dataset
    try:
        df = pd.read_csv('archive/cardio_train.csv', sep=';')
    except FileNotFoundError:
        print("Error: cardio_train.csv not found. Make sure it's in the 'archive/' directory.")
        return

    # Data Preprocessing
    df['age'] = (df['age'] / 365).round().astype(int)

    # Drop unnecessary columns
    df = df.drop(columns=['id'])

    # Define features and target
    X = df.drop(columns=['cardio'])
    y = df['cardio']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Heart Disease Model Accuracy: {accuracy:.4f}")

    # Save the model
    joblib.dump(model, 'heart_disease_model.joblib')
    print("Heart Disease Model saved as heart_disease_model.joblib")

if __name__ == '__main__':
    train_heart_disease_model()
