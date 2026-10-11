
import pandas as pd
from pathlib import Path

# Define project paths
project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "Dataset" / "SMSSpamCollection"
output_path = project_dir / "Dataset" / "spam_cleaned.csv"

# Load dataset
df = pd.read_csv(
    dataset_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("========== DATA CLEANING ==========")
print("Original dataset size:", len(df))

# Check for missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Remove rows with missing labels or messages, if any
df = df.dropna(subset=["label", "message"])

# Save cleaned dataset
df.to_csv(output_path, index=False)

print("\nCleaned dataset size:", len(df))
print("Duplicate rows remaining:", df.duplicated().sum())

print("\nCleaned label distribution:")
print(df["label"].value_counts())

print("\nCleaned dataset saved to:")
print(output_path)
