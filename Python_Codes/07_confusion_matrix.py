
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Load the cleaned dataset
project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "Dataset" / "spam_cleaned.csv"
results_dir = project_dir / "Results"
results_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(dataset_path)

X = df["message"].astype(str)
y = df["label"]

# Reproduce the same stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train the model using a pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
    )),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Generate confusion matrix
labels = ["ham", "spam"]
cm = confusion_matrix(y_test, predictions, labels=labels)

print("========== CONFUSION MATRIX ==========")
print(cm)
print("\nClassification report:")
print(classification_report(
    y_test, predictions,
    labels=labels,
    target_names=["Ham", "Spam"],
    zero_division=0
))

# Plot and save the chart
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

display.plot(values_format="d")
plt.title("Spam Detection - Confusion Matrix")
plt.tight_layout()

chart_path = results_dir / "confusion_matrix.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nChart saved to:", chart_path)
