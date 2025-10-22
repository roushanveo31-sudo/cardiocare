
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def train_cardio_model():
    """
    Trains and saves the cardiovascular disease prediction model.
    """
    # Load and preprocess the data
    df = pd.read_csv('archive/cardio_train.csv', sep=';')
    df = df.drop('id', axis=1)
    df['age'] = (df['age'] / 365).round().astype('int')
    X = df.drop('cardio', axis=1)
    y = df['cardio']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the scaler on the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the XGBoost model on the scaled training data
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_scaled, y_train)

    # Save the model and the scaler
    joblib.dump(xgb, 'cardio_model.joblib')
    joblib.dump(scaler, 'cardio_scaler.joblib')

    print('Cardiovascular disease model and scaler saved successfully.')

if __name__ == '__main__':
    train_cardio_model()
