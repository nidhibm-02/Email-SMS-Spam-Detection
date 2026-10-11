
import streamlit as st
import joblib
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "Results" / "spam_detection_model.joblib"

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Application heading
st.title("📩 SMS Spam Detection")
st.write(
    "Enter a text message below to check whether "
    "it is likely to be spam or legitimate."
)

# Message input
message = st.text_area(
    "Enter your message:",
    placeholder="Type or paste your message here..."
)

# Prediction button
if st.button("Check Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        try:
            model = load_model()

            prediction = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            classes = list(model.classes_)

            spam_index = classes.index("spam")
            spam_probability = probabilities[spam_index] * 100

            if prediction == "spam":
                st.error("🚨 This message is classified as SPAM.")
            else:
                st.success("✅ This message is classified as HAM (not spam).")

            st.metric(
                "Estimated spam probability",
                f"{spam_probability:.2f}%"
            )

            st.caption(
                "Predictions are estimates and may not always be correct."
            )

        except FileNotFoundError:
            st.error(
                "Saved model not found. Run 08_save_model.py first."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
