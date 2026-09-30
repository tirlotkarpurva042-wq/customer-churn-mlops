import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("=" * 50)
print("CUSTOMER CHURN MODEL MONITORING")
print("=" * 50)

# Load processed dataset
df = pd.read_csv("data/processed/processed_data.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Load trained model
model_package = joblib.load("models/churn_model.pkl")

model = model_package["model"]
scaler = model_package["scaler"]
numerical_columns = model_package["numerical_columns"]
feature_columns = model_package["feature_columns"]

# Prepare data
X = X[feature_columns]

X[numerical_columns] = scaler.transform(
    X[numerical_columns]
)

# Generate predictions
y_pred = model.predict(X)

# Calculate performance metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("\nMonitoring Results")
print("-" * 30)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Save monitoring results
monitoring_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

monitoring_results.to_csv(
    "monitoring_results.csv",
    index=False
)

print("\nMonitoring results saved successfully.")
print("File: monitoring_results.csv")