import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


DATASET_PATH = "dataset/bbc-news-data.csv"


def clean_text(text):
    """
    Clean news text before training or prediction.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_prepare_dataset():
    """
    Load the BBC News dataset, clean the text, and split it into train/test data.
    """
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, sep=None, engine="python")

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    text_column = "content"
    label_column = "category"

    print("\nChecking missing values:")
    print(df[[text_column, label_column]].isna().sum())

    df = df.dropna(subset=[text_column, label_column])

    print("\nCategory distribution:")
    print(df[label_column].value_counts())

    print("\nCleaning text...")
    df["clean_text"] = df[text_column].apply(clean_text)

    X = df["clean_text"]
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return X_train, X_test, y_train, y_test


def create_tfidf_features(X_train, X_test):
    """
    Convert text into TF-IDF numerical features.
    """
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 1),
        max_features=10000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Number of text features:", X_train_tfidf.shape[1])

    return vectorizer, X_train_tfidf, X_test_tfidf
