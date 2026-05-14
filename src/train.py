# Fraud Detection Model Training Script

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib


def load_data(path):
    """
    Load dataset from CSV file
    """
    data = pd.read_csv(path)
    return data


def preprocess_data(data):
    """
    Clean and encode dataset
    """

    # Drop unnecessary columns
    drop_cols = [
        "trans_date_trans_time",
        "first",
        "last",
        "street",
        "city",
        "state",
        "job",
        "dob",
        "trans_num"
    ]

    for col in drop_cols:
        if col in data.columns:
            data = data.drop(col, axis=1)

    # Convert categorical columns to numbers
    for column in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

    return data


def train_model(data):
    """
    Train Random Forest model for fraud detection
    """

    # Separate features and target
    X = data.drop("is_fraud", axis=1)
    y = data["is_fraud"]

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

    print("\nModel Performance:")
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

    # Load correct dataset
    data = load_data("data/fraudTrain.csv")

    # Preprocess data
    data = preprocess_data(data)

    # Train model
    model = train_model(data)

    # Save model
    save_model(model)

    print("Training complete.")
