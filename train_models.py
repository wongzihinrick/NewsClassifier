import os
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


DATASET_PATH = "dataset/bbc-news-data.csv"
SVM_MODEL_PATH = "models/svm_model.pkl"
LOGISTIC_MODEL_PATH = "models/logistic_regression_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
RESULTS_PATH = "models/model_comparison.csv"


def clean_text(text):
    """
    Clean news text before training.
    This keeps the text simple and consistent for NLP processing.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    print("NewsSort AI - Model Training and Comparison")
    print("=" * 50)

    print("\n1. Loading dataset...")
    df = pd.read_csv(DATASET_PATH, sep=None, engine="python")

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    text_column = "content"
    label_column = "category"

    print("\n2. Checking missing values...")
    print(df[[text_column, label_column]].isna().sum())

    df = df.dropna(subset=[text_column, label_column])

    print("\n3. Category distribution:")
    print(df[label_column].value_counts())

    print("\n4. Cleaning text...")
    df["clean_text"] = df[text_column].apply(clean_text)

    X = df["clean_text"]
    y = df[label_column]

    print("\n5. Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\n6. Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 1),
        max_features=10000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Number of text features:", X_train_tfidf.shape[1])

    print("\n7. Training and evaluating models...")
    models = {
        "Support Vector Machine": LinearSVC(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
            }
        )
        trained_models[model_name] = model

        print(f"\n{model_name} Results")
        print("-" * 30)
        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1-score :", round(f1, 4))
        print("\nDetailed classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    results_df = pd.DataFrame(results)
    print("\nModel Comparison")
    print("-" * 30)
    print(results_df.to_string(index=False))

    print("\n8. Saving models, vectorizer, and comparison results...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(trained_models["Support Vector Machine"], SVM_MODEL_PATH)
    joblib.dump(trained_models["Logistic Regression"], LOGISTIC_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("Saved SVM model to:", SVM_MODEL_PATH)
    print("Saved Logistic Regression model to:", LOGISTIC_MODEL_PATH)
    print("Saved vectorizer to:", VECTORIZER_PATH)
    print("Saved model comparison to:", RESULTS_PATH)
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
