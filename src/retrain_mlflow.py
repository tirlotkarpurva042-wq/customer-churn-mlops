import pandas as pd
import joblib
import mlflow
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("=" * 50)
print("RETRAINED MODEL - MLFLOW TRACKING")
print("=" * 50)

df = pd.read_csv("data/processed/processed_data.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numerical_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test_scaled[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

mlflow.set_experiment("Customer_Churn_Prediction")

with mlflow.start_run(run_name="Retrained_Logistic_Regression"):

    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("model_version", "v2")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "numerical_columns": numerical_columns,
            "feature_columns": X.columns.tolist()
        },
        "models/churn_model_v2.pkl"
    )

    mlflow.log_artifact(
        "models/churn_model_v2.pkl",
        artifact_path="model"
    )

print("\nRetrained model tracked successfully in MLflow.")

print("\nNew Model Performance")
print("-" * 30)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nModel saved:")
print("models/churn_model_v2.pkl")