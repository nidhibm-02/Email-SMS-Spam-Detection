
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load cleaned dataset
project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "Dataset" / "spam_cleaned.csv"

df = pd.read_csv(dataset_path)

X = df["message"].astype(str)
y = df["label"].map({"ham": 0, "spam": 1})

# Use the same split for a fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000
        )),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy (%)": round(
            accuracy_score(y_test, predictions) * 100, 2
        ),
        "Precision (%)": round(
            precision_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "Recall (%)": round(
            recall_score(y_test, predictions, zero_division=0) * 100, 2
        ),
        "F1-score (%)": round(
            f1_score(y_test, predictions, zero_division=0) * 100, 2
        )
    })

results_df = pd.DataFrame(results)

print("========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_path = project_dir / "Results" / "model_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\nComparison saved to:", results_path)
