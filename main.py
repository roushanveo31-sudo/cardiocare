
# main.py - Orchestrates the training of all three heart-related prediction models.

from heart_disease_model import train_heart_disease_model
from heart_attack_model import train_heart_attack_model
from heart_failure_model import train_heart_failure_model

if __name__ == "__main__":
    print("Starting the complete model training pipeline...")
    print("-" * 50)

    # Train the Heart Disease Model
    train_heart_disease_model()
    print("-" * 50)

    # Train the Heart Attack Risk Model
    train_heart_attack_model()
    print("-" * 50)

    # Train the Heart Failure Model
    train_heart_failure_model()
    print("-" * 50)

    print("All models have been trained successfully.")
