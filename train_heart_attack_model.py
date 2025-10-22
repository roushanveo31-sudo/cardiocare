
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
import numpy as np

def train_heart_attack_model():
    """
    Trains and saves the heart attack risk prediction model.
    """
    # Load the dataset
    df = pd.read_csv('archive (1)/heart_attack_prediction_india.csv')
    df = df.drop(['Patient_ID', 'State_Name'], axis=1)

    # Separate features and target
    X = df.drop('Heart_Attack_Risk', axis=1)
    y = df['Heart_Attack_Risk']

    # Preprocessing
    categorical_features = ['Gender']
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Gender' in numerical_features:
        numerical_features.remove('Gender')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )

    # Fit the preprocessor on the entire dataset
    X_processed = preprocessor.fit_transform(X)

    # Calculate class weights for the entire dataset
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict = dict(enumerate(class_weights))

    # Build the Neural Network
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_processed.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model on the entire dataset with class weights
    model.fit(X_processed, y, epochs=20, batch_size=32, verbose=0, class_weight=class_weights_dict)

    # Save the model and the preprocessor
    model.save('heart_attack_model.keras')
    joblib.dump(preprocessor, 'heart_attack_preprocessor.joblib')

    print('Heart attack risk model and preprocessor saved successfully.')

if __name__ == '__main__':
    train_heart_attack_model()
