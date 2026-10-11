# SMS Spam Detection Using Machine Learning

## Project Overview

This project classifies SMS text messages as **Spam** or **Ham (Not Spam)** using machine learning and Natural Language Processing (NLP). It uses TF-IDF to convert text into numerical features and a Multinomial Naive Bayes classifier to predict whether a message is likely to be spam.

The project also includes model comparison, performance evaluation, an interactive command-line predictor, and a web application developed using Streamlit.

## Application Preview

![SMS Spam Detection Web Application](screenshots/spam-detection-app.png)

## Features

- Dataset exploration and duplicate removal
- Text feature extraction using TF-IDF
- Machine learning model training and comparison
- Model evaluation using accuracy, precision, recall, F1-score, and a confusion matrix
- Saved-model predictions using Joblib
- Interactive command-line spam detection
- Web-based interface using Streamlit

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Logistic Regression
- Matplotlib
- Joblib
- Streamlit

## Dataset

The project uses the SMS Spam Collection dataset, which contains SMS messages labelled as spam or ham.

- Original dataset: 5,572 messages
- Duplicate records removed: 403
- Cleaned dataset: 5,169 messages
- Ham messages after cleaning: 4,516
- Spam messages after cleaning: 653

Dataset source: [UCI Machine Learning Repository — SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

## Machine Learning Workflow

1. Load and explore the dataset.
2. Remove duplicate records and save the cleaned dataset.
3. Split the data into training and testing sets.
4. Convert text messages into TF-IDF features.
5. Train a Multinomial Naive Bayes classifier.
6. Compare the results with Logistic Regression.
7. Evaluate the selected model using classification metrics and a confusion matrix.
8. Save the trained pipeline for reuse.
9. Build a Streamlit application for interactive predictions.

The training pipeline fits TF-IDF on the training data to avoid data leakage during evaluation.

## Model Performance

The Multinomial Naive Bayes model achieved the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 97.20% |
| Spam Precision | 100.00% |
| Spam Recall | 77.86% |
| Spam F1-score | 87.55% |

The model was also compared with Logistic Regression. Naive Bayes performed better on the reported test-set metrics in this experiment.

These results apply to the chosen dataset and test split; performance may differ on new messages.

## Project Structure

```text
Email_Spam_Detection/
├── Dataset/
│   └── spam_cleaned.csv
├── Python_Codes/
│   ├── 01_load_dataset.py
│   ├── 02_clean_dataset.py
│   ├── 03_tfidf_vectorization.py
│   ├── 04_train_model.py
│   ├── 05_compare_models.py
│   ├── 06_spam_predictor.py
│   ├── 07_confusion_matrix.py
│   ├── 08_save_model.py
│   ├── 09_predict_saved_model.py
│   └── 10_spam_web_app.py
├── Results/
│   ├── model_evaluation.txt
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   └── spam_detection_model.joblib
├── README.md
├── requirements.txt
└── .gitignore
```

*The file names shown above are based on the current project workflow. Keep the names that actually exist in your repository.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Email_Spam_Detection
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Prepare the model

If the saved model is included in the repository, make sure it is located at:

```text
Results/spam_detection_model.joblib
```

If it is not included, place the required dataset in the expected location and run:

```bash
python Python_Codes/08_save_model.py
```

### 4. Launch the web application

```bash
python -m streamlit run Python_Codes/10_spam_web_app.py
```

Open the local URL shown in the terminal to use the application.

## Example Predictions

**Example spam message**

> Congratulations! You have won a free prize. Claim now!

Expected classification: Spam

**Example normal message**

> Hi, are we still meeting at the library tomorrow?

Expected classification: Ham (Not Spam)

Predictions are estimates and are not guaranteed to be correct.

## Limitations

- The model was trained on SMS messages, so it is an SMS/text spam detector rather than a model specifically trained on email data.
- Spam messages may be missed, and legitimate messages may occasionally be misclassified.
- Performance on messages that differ substantially from the training dataset may vary.

## Future Enhancements

- Evaluate the model on additional message datasets.
- Improve the user interface and prediction explanations.
- Explore additional text classification algorithms.
- Deploy the Streamlit application online.

## Author

**Nidhi B M**

GitHub: [nidhibm-02](https://github.com/nidhibm-02)
