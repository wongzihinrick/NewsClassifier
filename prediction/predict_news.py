import argparse
import re
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "svm": PROJECT_ROOT / "models" / "svm_model.pkl",
    "logistic": PROJECT_ROOT / "models" / "logistic_regression_model.pkl",
}
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"


def clean_text(text):
    """
    Clean the input news text using the same style as the training file.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_category(news_text, model_name="svm"):
    """
    Load the saved model pipeline, then predict the news category.
    """
    model = joblib.load(MODEL_PATHS[model_name])
    vectorizer = joblib.load(VECTORIZER_PATH) if VECTORIZER_PATH.exists() else None

    cleaned_text = clean_text(news_text)
    if hasattr(model, "named_steps"):
        prediction = model.predict([cleaned_text])[0]
    else:
        text_features = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_features)[0]

    return prediction


def main():
    parser = argparse.ArgumentParser(description="Predict a BBC news category.")
    parser.add_argument(
        "--model",
        choices=MODEL_PATHS.keys(),
        default="svm",
        help="Choose the trained model to use for prediction.",
    )
    args = parser.parse_args()

    print("NewsSort AI - Test Prediction")
    print("=" * 35)
    print("Prediction model:", args.model)
    print("Enter a news title or short article below.")
    print("Type 'exit' to stop.\n")

    while True:
        news_text = input("News text: ")

        if news_text.lower().strip() == "exit":
            print("Goodbye.")
            break

        if not news_text.strip():
            print("Please enter some news text.\n")
            continue

        category = predict_category(news_text, args.model)
        print("Predicted category:", category)
        print()


if __name__ == "__main__":
    main()
