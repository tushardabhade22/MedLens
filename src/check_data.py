import pandas as pd

# Load dataset
df = pd.read_csv("../data/reports.csv")

# Display first 5 rows
print(df.head())

# Dataset size
print("\nDataset shape:", df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Count samples in each category
print("\nCategory distribution:")
print(df["label"].value_counts())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())