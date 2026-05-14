from fastapi import FastAPI
import joblib
import os
import pandas as pd
from pydantic import BaseModel

# Start FastAPI
app = FastAPI()

# Load trained model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)


# Define full input structure
class Transaction(BaseModel):
    Unnamed_0: int
    cc_num: float
    merchant: int
    category: int
    amt: float
    gender: int
    zip: int
    lat: float
    long: float
    city_pop: int
    unix_time: int
    merch_lat: float
    merch_long: float


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):

    # Create dataframe with EXACT feature names expected
    data = pd.DataFrame([{
        "Unnamed: 0": transaction.Unnamed_0,
        "cc_num": transaction.cc_num,
        "merchant": transaction.merchant,
        "category": transaction.category,
        "amt": transaction.amt,
        "gender": transaction.gender,
        "zip": transaction.zip,
        "lat": transaction.lat,
        "long": transaction.long,
        "city_pop": transaction.city_pop,
        "unix_time": transaction.unix_time,
        "merch_lat": transaction.merch_lat,
        "merch_long": transaction.merch_long
    }])

    # Predict
    prediction = model.predict(data)[0]

    # Return readable result
    if prediction == 1:
        return {"prediction": "Fraud"}
    else:
        return {"prediction": "Not Fraud"}
