import pandas as pd

print("=" * 50)
print("SIMULATED MODEL PERFORMANCE DROP")
print("=" * 50)

# Load current monitoring results
results = pd.read_csv("monitoring_results.csv")

print("\nCurrent Performance:")
print(results)

# Simulate performance drop
results.loc[results["Metric"] == "Accuracy", "Value"] = 0.65
results.loc[results["Metric"] == "Precision", "Value"] = 0.45
results.loc[results["Metric"] == "Recall", "Value"] = 0.40
results.loc[results["Metric"] == "F1 Score", "Value"] = 0.42

results.to_csv(
    "monitoring_results_degraded.csv",
    index=False
)

print("\nSimulated degraded performance:")
print(results)

print("\nPerformance drop simulation completed.")
print("File: monitoring_results_degraded.csv")
