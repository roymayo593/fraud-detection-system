# Fraud Detection Prediction Script

import joblib
import numpy as np


def load_model(path="model.pkl"):
    """
    Load trained model from file
    """
    model = joblib.load(path)
    return model


def predict_transaction(model, transaction):
    """
    Predict whether a transaction is fraud or not
    """
    transaction = np.array(transaction).reshape(1, -1)
    prediction = model.predict(transaction)

    if prediction[0] == 1:
        return "Fraud"
    else:
        return "Not Fraud"


if __name__ == "__main__":
    print("Loading model...")

    model = load_model()

    # Example transaction (dummy values)
    sample_transaction = [0.1] * 30

    result = predict_transaction(model, sample_transaction)

    print(f"Prediction: {result}")
