# Fraud Detection Model Training Script

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


def load_data(path):
    """
    Load dataset from CSV file
    """
    data = pd.read_csv(path)
    return data


def train_model(data):
    """
    Train Random Forest model for fraud detection
    """
    # Separate features and target
    X = data.drop("Class", axis=1)
    y = data["Class"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)

    print("Model Performance:")
    print(classification_report(y_test, predictions))

    return model


def save_model(model, path="model.pkl"):
    """
    Save trained model to file
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    print("Starting fraud detection training...")

    # NOTE: dataset will be added later
    data = load_data("data/creditcard.csv")

    model = train_model(data)

    save_model(model)

    print("Training complete.")
