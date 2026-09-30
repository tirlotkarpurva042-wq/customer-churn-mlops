import pandas as pd

print("=" * 50)
print("MODEL PERFORMANCE CHECK")
print("=" * 50)

# Load monitoring results
results = pd.read_csv("monitoring_results_degraded.csv")

# Get F1 score
f1_score = results.loc[
    results["Metric"] == "F1 Score",
    "Value"
].iloc[0]

print(f"\nCurrent F1 Score: {f1_score:.2f}")

# Set performance threshold
threshold = 0.50

print(f"Retraining Threshold: {threshold:.2f}")

if f1_score < threshold:
    print("\nWARNING: Model performance has dropped.")
    print("Retraining is required.")
else:
    print("\nModel performance is acceptable.")
    print("Retraining is not required.")