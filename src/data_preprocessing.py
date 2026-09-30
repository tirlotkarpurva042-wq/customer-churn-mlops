import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "data/raw/customer_churn.csv"
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# Remove customer ID because it is only an identifier
df = df.drop("customerID", axis=1)

# Convert TotalCharges from string to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values created during conversion
print("\nMissing values after converting TotalCharges:")
print(df["TotalCharges"].isnull().sum())

# Fill missing TotalCharges using the median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Convert target variable
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# Convert categorical columns using one-hot encoding
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Convert boolean columns to integers
X = X.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nProcessed feature shape:", X.shape)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTarget distribution:")
print(y.value_counts())

# Save processed data
processed_df = X.copy()
processed_df["Churn"] = y

processed_df.to_csv(
    "data/processed/processed_data.csv",
    index=False
)

print("\nProcessed dataset saved successfully.")