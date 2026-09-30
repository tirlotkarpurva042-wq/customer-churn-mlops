from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0"
)

# Load trained model
model_package = joblib.load("models/churn_model_v2.pkl")

model = model_package["model"]
scaler = model_package["scaler"]
numerical_columns = model_package["numerical_columns"]
feature_columns = model_package["feature_columns"]


# Input data structure
class CustomerData(BaseModel):
    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    tenure: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Convert input into DataFrame
    data = pd.DataFrame([customer.model_dump()])

    # Rename columns to match training dataset
    data = data.rename(columns={
        "senior_citizen": "SeniorCitizen",
        "monthly_charges": "MonthlyCharges",
        "total_charges": "TotalCharges",
        "gender": "gender",
        "partner": "Partner",
        "dependents": "Dependents",
        "phone_service": "PhoneService",
        "multiple_lines": "MultipleLines",
        "internet_service": "InternetService",
        "online_security": "OnlineSecurity",
        "online_backup": "OnlineBackup",
        "device_protection": "DeviceProtection",
        "tech_support": "TechSupport",
        "streaming_tv": "StreamingTV",
        "streaming_movies": "StreamingMovies",
        "contract": "Contract",
        "paperless_billing": "PaperlessBilling",
        "payment_method": "PaymentMethod"
    })

    # One-hot encode categorical columns
    categorical_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    data = pd.get_dummies(
        data,
        columns=categorical_columns,
        drop_first=True
    )

    # Add missing training columns
    for column in feature_columns:
        if column not in data.columns:
            data[column] = 0

    # Remove unexpected columns and maintain correct order
    data = data[feature_columns]

    # Convert boolean values to integers
    data = data.astype(int)

    # Scale numerical columns
    data[numerical_columns] = scaler.transform(
        data[numerical_columns]
    )

    # Make prediction
    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    # Convert prediction to readable result
    if prediction == 1:
        result = "Churn"
    else:
        result = "No Churn"

    return {
        "prediction": result,
        "churn_probability": round(float(probability), 4)
    }