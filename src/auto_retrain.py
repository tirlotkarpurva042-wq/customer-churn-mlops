import pandas as pd
import subprocess
import sys

print("=" * 50)
print("AUTOMATIC MODEL RETRAINING SYSTEM")
print("=" * 50)

# Read current monitoring results
results = pd.read_csv("monitoring_results_degraded.csv")

# Get F1 score
f1_score = results.loc[
    results["Metric"] == "F1 Score",
    "Value"
].iloc[0]

# Retraining threshold
threshold = 0.50

print(f"\nCurrent F1 Score : {f1_score:.2f}")
print(f"Retraining Threshold: {threshold:.2f}")

# Check model performance
if f1_score < threshold:

    print("\nWARNING: Model performance is below threshold.")
    print("Automatic retraining is being triggered...")

    # Automatically run retraining + MLflow tracking
    subprocess.run(
        [sys.executable, "src/retrain_mlflow.py"],
        check=True
    )

    print("\nAutomatic retraining completed successfully.")
    print("New model created: models/churn_model_v2.pkl")

else:

    print("\nModel performance is acceptable.")
    print("Automatic retraining is not required.")