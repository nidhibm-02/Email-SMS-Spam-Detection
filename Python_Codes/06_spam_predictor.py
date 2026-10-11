
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Locate and load the cleaned dataset
project_dir = Path(__file__).resolve().parent.parent
dataset_path = project_dir / "Dataset" / "spam_cleaned.csv"

df = pd.read_csv(dataset_path)

X = df["message"].astype(str)
y = df["label"]

# Build and train the complete text-classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
    )),
    ("classifier", MultinomialNB())
])

model.fit(X, y)

print("========== EMAIL / MESSAGE SPAM PREDICTOR ==========")
print("Enter 'exit' to close the program.")

while True:
    message = input("\nEnter your message: ").strip()

    if message.lower() == "exit":
        print("Predictor closed.")
        break

    if not message:
        print("Please enter a message.")
        continue

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = list(model.named_steps["classifier"].classes_)
    spam_probability = probabilities[classes.index("spam")]

    print("\nPrediction:", prediction.upper())
    print(f"Estimated spam probability: {spam_probability:.2%}")

    if prediction == "spam":
        print("Result: This message resembles spam examples.")
    else:
        print("Result: This message resembles legitimate examples.")
