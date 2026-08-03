import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "Support Vector Machine": PROJECT_ROOT / "models" / "svm_model.pkl",
    "Logistic Regression": PROJECT_ROOT / "models" / "logistic_regression_model.pkl",
}
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
RESULTS_PATH = PROJECT_ROOT / "results" / "model_comparison.csv"


def clean_text(text):
    """
    Clean the input news text using the same style as the training file.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_resource
def load_model_files():
    """
    Load the saved model pipelines.
    """
    models = {}
    for model_name, model_path in MODEL_PATHS.items():
        models[model_name] = joblib.load(model_path)

    vectorizer = joblib.load(VECTORIZER_PATH) if VECTORIZER_PATH.exists() else None
    return models, vectorizer


def load_model_comparison():
    """
    Load model comparison results if the file exists.
    """
    try:
        return pd.read_csv(RESULTS_PATH)
    except FileNotFoundError:
        return None


def predict_category(news_text, selected_model_name):
    """
    Predict the category of the news text using the selected model.
    """
    models, vectorizer = load_model_files()
    model = models[selected_model_name]

    cleaned_text = clean_text(news_text)
    if hasattr(model, "named_steps"):
        prediction = model.predict([cleaned_text])[0]
    else:
        text_features = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_features)[0]

    return prediction


def main():
    st.set_page_config(
        page_title="NewsSort AI",
        page_icon="N",
        layout="centered",
    )

    st.title("NewsSort AI")
    st.subheader("News Category Classification System Using NLP")

    st.write(
        "Enter a news title or article text below. "
        "The system will predict whether it belongs to business, entertainment, politics, sport, or tech."
    )

    with st.form("prediction_form", border=True):
        selected_model_name = st.selectbox(
            "Choose prediction model",
            list(MODEL_PATHS.keys()),
        )

        st.info(f"Current prediction model: Word + Character TF-IDF + {selected_model_name}")

        news_text = st.text_area(
            "News text",
            height=180,
            placeholder="Example: Apple launches new artificial intelligence features for iPhone users...",
        )

        submitted = st.form_submit_button("Predict category", icon=":material/search:")

    if submitted:
        if not news_text.strip():
            st.warning("Please enter some news text first.")
            return

        prediction = predict_category(news_text, selected_model_name)
        st.success(f"Predicted Category: {prediction.title()}")

        st.caption(
            "Note: This prototype is for academic demonstration only. "
            "It predicts a category based on patterns learned from the training dataset."
        )

    st.divider()
    st.subheader("Model Comparison")

    comparison_df = load_model_comparison()
    if comparison_df is None:
        st.warning("Model comparison file not found. Please run the training scripts first.")
    else:
        display_df = comparison_df.copy()
        metric_columns = ["Accuracy", "Precision", "Recall", "F1-score"]

        for column in metric_columns:
            if column in display_df.columns:
                display_df[column] = (display_df[column] * 100).round(2)

        st.dataframe(display_df, hide_index=True)
        st.caption("Scores are shown as percentages.")

    with st.expander("About this system"):
        st.write(
            """
            NewsSort AI uses Natural Language Processing to classify news text.

            Current workflow:
            1. Clean the news text.
            2. Convert text into numerical features using word and character TF-IDF.
            3. Train and compare Support Vector Machine and Logistic Regression.
            4. Allow users to choose between Support Vector Machine and Logistic Regression.
            """
        )


if __name__ == "__main__":
    main()
