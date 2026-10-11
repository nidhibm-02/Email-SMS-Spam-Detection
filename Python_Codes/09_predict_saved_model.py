
import joblib
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
model_path = project_dir / "Results" / "spam_detection_model.joblib"

# Load the previously trained pipeline
model = joblib.load(model_path)

print("========== SAVED SPAM DETECTOR ==========")
print("Enter 'exit' to stop.")

while True:
    message = input("\nEnter a message: ").strip()

    if message.lower() == "exit":
        print("Program closed.")
        break

    if not message:
        print("Please enter a message.")
        continue

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = list(model.named_steps["classifier"].classes_)
    spam_probability = probabilities[classes.index("spam")]

    print("Prediction:", prediction.upper())
    print(f"Estimated spam probability: {spam_probability:.2%}")
