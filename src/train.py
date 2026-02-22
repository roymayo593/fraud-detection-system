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
    Basic preprocessing:
    - Drop unnecessary columns
    - Encode categorical variables
    """
    # Drop ID-like columns if present
    drop_cols = ["trans_date_trans_time", "merchant", "first", "last",
                 "street", "city", "state", "zip", "job", "dob"]
    
    for col in drop_cols:
        if col in data.columns:
            data = data.drop(col, axis=1)

    # Encode categorical columns
    label_encoders = {}
    for column in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    return data


def train_model(data):
    """
    Train Random Forest model
    """
    X = data.drop("is_fraud", axis=1)
    y = data["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Model Performance:")
    print(classification_report(y_test, predictions))

    return model


def save_model(model, path="model.pkl"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    print("Loading dataset...")
    data = load_data("data/fraudTrain.csv")

    print("Preprocessing data...")
    data = preprocess_data(data)

    print("Training model...")
    model = train_model(data)

    print("Saving model...")
    save_model(model)

    print("Training complete.")

