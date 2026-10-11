
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Define paths
project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "Dataset" / "spam_cleaned.csv"
output_path = project_dir / "Results" / "tfidf_feature_info.txt"

# Load cleaned dataset
df = pd.read_csv(dataset_path)

X_text = df["message"].astype(str)
y = df["label"]

# Convert text into numerical TF-IDF features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(X_text)

print("========== TF-IDF VECTORIZATION ==========")
print("Number of messages:", X_tfidf.shape[0])
print("Number of numerical features:", X_tfidf.shape[1])
print("Feature matrix shape:", X_tfidf.shape)

print("\nFirst 20 vocabulary terms:")
print(vectorizer.get_feature_names_out()[:20])

print("\nFirst message:")
print(X_text.iloc[0])

print("\nFirst message TF-IDF vector:")
print(X_tfidf[0])

# Save basic feature information
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as file:
    file.write("TF-IDF FEATURE INFORMATION\n")
    file.write(f"Messages: {X_tfidf.shape[0]}\n")
    file.write(f"Features: {X_tfidf.shape[1]}\n")
    file.write(f"Matrix shape: {X_tfidf.shape}\n")

print("\nFeature information saved to:", output_path)
