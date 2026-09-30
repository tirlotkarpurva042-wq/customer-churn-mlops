import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load processed dataset
df = pd.read_csv("data/processed/processed_data.csv")

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create MLflow experiment
mlflow.set_experiment("Customer_Churn_Prediction")


# ==================================================
# 1. LOGISTIC REGRESSION
# ==================================================

with mlflow.start_run(run_name="Logistic_Regression"):

    # Numerical columns
    numerical_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    # Scale numerical features
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_columns] = scaler.fit_transform(
        X_train[numerical_columns]
    )

    X_test_scaled[numerical_columns] = scaler.transform(
        X_test[numerical_columns]
    )

    # Create model
    model = LogisticRegression(max_iter=1000)

    # Train model
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("random_state", 42)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save Logistic Regression model
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "numerical_columns": numerical_columns,
            "feature_columns": X.columns.tolist()
        },
        "models/logistic_regression_mlflow.pkl"
    )

    # Log model file to MLflow
    mlflow.log_artifact(
        "models/logistic_regression_mlflow.pkl",
        artifact_path="model"
    )

    print("Logistic Regression MLflow run completed.")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


# ==================================================
# 2. RANDOM FOREST
# ==================================================

with mlflow.start_run(run_name="Random_Forest"):

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.20)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save Random Forest model
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/random_forest_mlflow.pkl"
    )

    # Log saved model file to MLflow
    mlflow.log_artifact(
        "models/random_forest_mlflow.pkl",
        artifact_path="model"
    )

    print("\nRandom Forest MLflow run completed.")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


# ==================================================
# COMPLETION
# ==================================================

print("\nMLflow experiment completed successfully.")