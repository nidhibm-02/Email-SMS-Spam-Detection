
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Define project paths
project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "Dataset" / "spam_cleaned.csv"
results_dir = project_dir / "Results"
results_dir.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(dataset_path)

X = df["message"].astype(str)
y = df["label"].map({"ham": 0, "spam": 1})

if y.isnull().any():
    raise ValueError("Unexpected label found in the dataset.")

# Split data before fitting the vectorizer
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Learn vocabulary from training messages only
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on unseen test messages
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("========== SPAM DETECTION MODEL ==========")
print("Algorithm: Multinomial Naive Bayes")
print("Total messages:", len(df))
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

print(f"\nAccuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Save the evaluation results
report_path = results_dir / "model_evaluation.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("EMAIL SPAM DETECTION - MODEL EVALUATION\n")
    file.write("Algorithm: Multinomial Naive Bayes\n")
    file.write(f"Training messages: {len(X_train)}\n")
    file.write(f"Testing messages: {len(X_test)}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n")
    file.write("\nClassification Report:\n")
    file.write(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Ham", "Spam"],
            zero_division=0
        )
    )
    file.write("\nConfusion Matrix:\n")
    file.write(
        str(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    )

print("\nEvaluation report saved to:", report_path)
