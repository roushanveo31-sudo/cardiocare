
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_heart_disease_model():
    """
    Trains and saves the heart disease diagnosis model.
    """
    # Load the dataset
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    df = pd.read_csv('heart+disease/processed.cleveland.data', names=column_names, na_values='?')

    # Impute missing values
    for col in ['ca', 'thal']:
        df[col].fillna(df[col].median(), inplace=True)

    # Binarize the target variable
    df['num'] = (df['num'] > 0).astype(int)

    # Separate features and target
    X = df.drop('num', axis=1)
    y = df['num']

    # Fit the scaler on the entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the Logistic Regression model on the entire dataset
    lr = LogisticRegression(random_state=42)
    lr.fit(X_scaled, y)

    # Save the model and the scaler
    joblib.dump(lr, 'heart_disease_model.joblib')
    joblib.dump(scaler, 'heart_disease_scaler.joblib')

    print('Heart disease diagnosis model and scaler saved successfully.')

if __name__ == '__main__':
    train_heart_disease_model()
