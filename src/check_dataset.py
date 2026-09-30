import pandas as pd

# Load the raw dataset
file_path = "data/raw/customer_churn.csv"
df = pd.read_csv(file_path)

print("=" * 50)
print("CUSTOMER CHURN DATASET INFORMATION")
print("=" * 50)

# Dataset size
print("\n1. Dataset Shape:")
print(df.shape)

# Column names
print("\n2. Column Names:")
print(df.columns.tolist())

# First 5 rows
print("\n3. First 5 Rows:")
print(df.head())

# Data types
print("\n4. Data Types:")
print(df.dtypes)

# Missing values
print("\n5. Missing Values:")
print(df.isnull().sum())

# Churn distribution
print("\n6. Churn Distribution:")
print(df["Churn"].value_counts())

# Churn percentage
print("\n7. Churn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Basic numerical statistics
print("\n8. Numerical Statistics:")
print(df.describe())

print("\n" + "=" * 50)
print("DATASET INSPECTION COMPLETED")
print("=" * 50)