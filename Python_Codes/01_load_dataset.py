
import pandas as pd
from pathlib import Path

# Get the project folder path
project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "Dataset" / "SMSSpamCollection"

# Load the dataset
df = pd.read_csv(
    dataset_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Display dataset information
print("========== DATASET SUMMARY ==========")
print("Total messages:", len(df))

print("\nFirst five messages:")
print(df.head())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())
