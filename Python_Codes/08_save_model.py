
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Project paths
project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "Dataset" / "spam_cleaned.csv"
results_dir = project_dir / "Results"
results_dir.mkdir(parents=True, exist_ok=True)

model_path = results_dir / "spam_detection_model.joblib"

# Load cleaned data
df = pd.read_csv(dataset_path)
X = df["message"].astype(str)
y = df["label"]

# Create the same reproducible split used earlier
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create the complete pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
    )),
    ("classifier", MultinomialNB())
])

# Train using only the training data
model.fit(X_train, y_train)

# Evaluate on held-out test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save the complete pipeline
joblib.dump(model, model_path)

print("========== MODEL SAVING ==========")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print("Model saved to:", model_path)
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))
