import re

import joblib


MODEL_PATH = "models/svm_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def clean_text(text):
    """
    Clean the input news text using the same style as the training file.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_category(news_text):
    """
    Load the saved SVM model and TF-IDF vectorizer, then predict the news category.
    """
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    cleaned_text = clean_text(news_text)
    text_features = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_features)[0]

    return prediction


def main():
    print("NewsSort AI - Test Prediction")
    print("=" * 35)
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

        category = predict_category(news_text)
        print("Predicted category:", category)
        print()


if __name__ == "__main__":
    main()
