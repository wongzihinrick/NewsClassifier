import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "Support Vector Machine": PROJECT_ROOT / "models" / "svm_model.pkl",
    "Logistic Regression": PROJECT_ROOT / "models" / "logistic_regression_model.pkl",
    "Multinomial Naive Bayes": PROJECT_ROOT / "models" / "multinomial_naive_bayes_model.pkl",
}
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
RESULTS_PATH = PROJECT_ROOT / "results" / "model_comparison.csv"

QUICK_SAMPLES = {
    "Tech": "Apple announced new artificial intelligence features for mobile devices, software developers, and cloud technology users.",
    "Business": "The company reported higher profit after strong sales growth, market recovery, and increased investor confidence.",
    "Politics": "The government announced a new policy after parliament debated election reform and public service funding.",
    "Sport": "The football team won the final match after the player scored a late goal in the tournament.",
    "Entertainment": "The actor received an award at the film festival after the movie became popular with audiences.",
}


def apply_professional_news_blue_theme():
    """
    Apply a clean light dashboard theme using professional news-blue colors.
    """
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #F8FAFC;
                --card-bg: #FFFFFF;
                --primary-blue: #2563EB;
                --primary-blue-dark: #1D4ED8;
                --navy: #0F172A;
                --muted-text: #64748B;
                --border: #E2E8F0;
                --success: #16A34A;
                --warning: #DC2626;
            }

            .stApp {
                background: var(--app-bg);
                color: var(--navy);
            }

            h1, h2, h3, h4, h5, h6, p, label, span, div {
                color: var(--navy);
            }

            [data-testid="stSidebar"] {
                background: var(--navy);
                border-right: 1px solid #1E293B;
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] div {
                color: #F8FAFC;
            }

            [data-testid="stSidebar"] .stCaptionContainer,
            [data-testid="stSidebar"] small {
                color: #CBD5E1;
            }

            [data-testid="stSidebar"] button {
                background: #1E293B;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 10px;
            }

            [data-testid="stSidebar"] button:hover {
                background: var(--primary-blue);
                color: #FFFFFF;
                border-color: var(--primary-blue);
            }

            div[data-testid="stForm"],
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 16px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            }

            textarea,
            input,
            [data-baseweb="select"] > div {
                background: #FFFFFF !important;
                color: var(--navy) !important;
                border-color: var(--border) !important;
            }

            textarea:focus,
            input:focus {
                border-color: var(--primary-blue) !important;
                box-shadow: 0 0 0 1px var(--primary-blue) !important;
            }

            .stButton > button {
                border-radius: 10px;
                border: 1px solid var(--primary-blue);
                background: var(--primary-blue);
                color: #FFFFFF;
                font-weight: 600;
            }

            .stButton > button:hover {
                background: var(--primary-blue-dark);
                color: #FFFFFF;
                border-color: var(--primary-blue-dark);
            }

            [data-testid="stMetric"] {
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 18px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            }

            [data-testid="stMetricLabel"] p {
                color: var(--muted-text);
                font-weight: 700;
            }

            [data-testid="stMetricValue"] {
                color: var(--primary-blue);
            }

            [data-testid="stDataFrame"] {
                background: var(--card-bg);
                border-radius: 14px;
            }

            .stAlert {
                border-radius: 12px;
            }

            hr {
                border-color: var(--border);
            }

            .stCaptionContainer {
                color: var(--muted-text);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


@st.cache_data
def load_model_comparison():
    """
    Load model comparison results if the file exists.
    """
    try:
        return pd.read_csv(RESULTS_PATH)
    except FileNotFoundError:
        return None


def get_best_model(comparison_df):
    """
    Return the model with the highest F1-score from the comparison results.
    """
    if comparison_df is None or comparison_df.empty or "F1-score" not in comparison_df.columns:
        return None

    best_row = comparison_df.sort_values("F1-score", ascending=False).iloc[0]
    return best_row["Model"], best_row["F1-score"]


def get_model_parts(model, vectorizer):
    """
    Return the feature extractor and classifier from a saved model.
    """
    if hasattr(model, "named_steps"):
        return model.named_steps["features"], model.named_steps["model"]

    return vectorizer, model


def predict_category(news_text, selected_model_name):
    """
    Predict the category of the news text using the selected model.
    """
    models, vectorizer = load_model_files()
    model = models[selected_model_name]
    cleaned_text = clean_text(news_text)

    if hasattr(model, "named_steps"):
        return model.predict([cleaned_text])[0]

    text_features = vectorizer.transform([cleaned_text])
    return model.predict(text_features)[0]


def explain_prediction(news_text, selected_model_name):
    """
    Return model scores and detected TF-IDF terms for the input text.
    """
    models, vectorizer = load_model_files()
    model = models[selected_model_name]
    cleaned_text = clean_text(news_text)
    feature_extractor, classifier = get_model_parts(model, vectorizer)

    text_features = feature_extractor.transform([cleaned_text])

    if hasattr(classifier, "predict_proba"):
        raw_scores = classifier.predict_proba(text_features)[0]
        score_label = "Probability"
    else:
        raw_scores = classifier.decision_function(text_features)[0]
        score_label = "Decision score"

    score_df = pd.DataFrame(
        {
            "Category": classifier.classes_,
            score_label: raw_scores,
        }
    ).sort_values(score_label, ascending=False)

    feature_names = feature_extractor.get_feature_names_out()
    feature_row = text_features.tocsr()[0]
    top_feature_indices = feature_row.indices[feature_row.data.argsort()[::-1]][:10]
    detected_terms = [
        feature_names[index]
        .replace("word_tfidf__", "")
        .replace("char_tfidf__", "")
        for index in top_feature_indices
    ]

    return score_df, detected_terms, score_label


def get_top_score(score_df, score_label):
    """
    Convert the top model score into a display-friendly value.
    """
    top_row = score_df.iloc[0]
    top_value = top_row[score_label]

    if score_label == "Probability":
        return f"{top_value * 100:.2f}%"

    return f"{top_value:.4f}"


def prepare_metrics_table(comparison_df):
    """
    Format model evaluation metrics as percentages for display.
    """
    if comparison_df is None:
        return None

    display_df = comparison_df.copy()
    metric_columns = ["Accuracy", "Precision", "Recall", "F1-score"]

    for column in metric_columns:
        if column in display_df.columns:
            display_df[column] = (display_df[column] * 100).round(2)

    return display_df


def set_quick_sample(sample_name):
    """
    Put a quick test sample into the text box.
    """
    st.session_state["news_text"] = QUICK_SAMPLES[sample_name]


def render_sidebar():
    """
    Create the sidebar navigation and controls.
    """
    st.sidebar.title("NewsSort AI")
    st.sidebar.caption("NLP News Category Classifier")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Predict News Category",
            "Multi-Model Comparison",
            "Model Evaluation Results",
            "About System",
        ],
    )

    st.sidebar.divider()
    selected_model_name = st.sidebar.selectbox(
        "Primary Model",
        list(MODEL_PATHS.keys()),
    )

    st.sidebar.divider()
    st.sidebar.write("Quick Test Samples")
    sample_cols = st.sidebar.columns(2)
    sample_names = list(QUICK_SAMPLES.keys())
    for index, sample_name in enumerate(sample_names):
        with sample_cols[index % 2]:
            st.button(
                sample_name,
                use_container_width=True,
                on_click=set_quick_sample,
                args=(sample_name,),
            )

    st.sidebar.divider()
    st.sidebar.write("Pipeline Details")
    st.sidebar.caption("Dataset: BBC News")
    st.sidebar.caption("Input column: content")
    st.sidebar.caption("Label column: category")
    st.sidebar.caption("Vectorizer: Word + Character TF-IDF")
    st.sidebar.caption("Models: SVM, Logistic Regression, Naive Bayes")

    return page, selected_model_name


def render_news_input():
    """
    Render shared news text input area.
    """
    if "news_text" not in st.session_state:
        st.session_state["news_text"] = ""

    return st.text_area(
        "News Article Input",
        key="news_text",
        height=220,
        placeholder="Type or paste news text here. Example: Apple launches new artificial intelligence features for iPhone users...",
    )


def render_score_breakdown(score_df, score_label, selected_model_name):
    """
    Show score/probability table and chart for one selected model.
    """
    st.subheader("Confidence / Score Breakdown")

    if score_label == "Probability":
        display_scores = score_df.copy()
        display_scores["Probability"] = (display_scores["Probability"] * 100).round(2)
        st.dataframe(
            display_scores,
            column_config={
                "Probability": st.column_config.NumberColumn(
                    "Probability",
                    format="%.2f%%",
                ),
            },
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(display_scores.set_index("Category"), y="Probability")
        st.caption(f"{selected_model_name} chooses the category with the highest probability.")
    else:
        display_scores = score_df.copy()
        display_scores["Decision score"] = display_scores["Decision score"].round(4)
        st.dataframe(display_scores, hide_index=True, use_container_width=True)
        st.bar_chart(display_scores.set_index("Category"), y="Decision score")
        st.caption(
            f"{selected_model_name} chooses the category with the strongest decision score. "
            "For SVM, this is a decision score rather than a true probability."
        )


def render_detected_terms(detected_terms):
    """
    Show important TF-IDF signals found in the input text.
    """
    if not detected_terms:
        return

    st.subheader("Detected TF-IDF Signals")
    st.write(", ".join(detected_terms))
    st.caption("These are strong text signals detected after TF-IDF feature extraction.")


def render_predict_page(selected_model_name):
    """
    Single-model prediction page.
    """
    st.title("News Article Classifier")
    st.write(
        "Enter a news title or article text below. "
        "The system will predict whether it belongs to business, entertainment, politics, sport, or tech."
    )

    news_text = render_news_input()
    col1, col2 = st.columns([1, 1])
    predict_clicked = col1.button("Predict Category", type="primary", use_container_width=True)
    clear_clicked = col2.button("Clear Input", use_container_width=True)

    if clear_clicked:
        st.session_state["news_text"] = ""
        st.rerun()

    if predict_clicked:
        if len(news_text.split()) < 10:
            st.warning("Please enter at least 10 words so the model has enough text to classify.")
            return

        prediction = predict_category(news_text, selected_model_name)
        score_df, detected_terms, score_label = explain_prediction(news_text, selected_model_name)
        top_score = get_top_score(score_df, score_label)

        st.success(f"Predicted Category: {prediction.title()} ({score_label}: {top_score})")

        with st.container(border=True):
            render_score_breakdown(score_df, score_label, selected_model_name)
            render_detected_terms(detected_terms)

        st.caption(
            "Note: This prototype is for academic demonstration only. "
            "It predicts a category based on patterns learned from the training dataset."
        )


def render_multi_model_page():
    """
    Show all model predictions for the same input text.
    """
    st.title("Live Multi-Model Prediction Comparison")
    st.write("Use this page to compare how all three trained models classify the same news text.")

    news_text = render_news_input()
    col1, col2 = st.columns([1, 1])
    compare_clicked = col1.button("Compare All Models", type="primary", use_container_width=True)
    clear_clicked = col2.button("Clear Input", use_container_width=True)

    if clear_clicked:
        st.session_state["news_text"] = ""
        st.rerun()

    if compare_clicked:
        if len(news_text.split()) < 10:
            st.warning("Please enter at least 10 words so the models have enough text to classify.")
            return

        comparison_df = load_model_comparison()
        best_model = get_best_model(comparison_df)
        best_model_name = best_model[0] if best_model else None

        prediction_rows = []
        for model_name in MODEL_PATHS:
            prediction = predict_category(news_text, model_name)
            score_df, _, score_label = explain_prediction(news_text, model_name)
            top_score = get_top_score(score_df, score_label)

            if model_name == best_model_name:
                status = "Best model by F1-score"
            else:
                status = "Prediction agreement"

            prediction_rows.append(
                {
                    "Model": model_name,
                    "Prediction": prediction.title(),
                    "Score Type": score_label,
                    "Top Score": top_score,
                    "Status": status,
                }
            )

        prediction_df = pd.DataFrame(prediction_rows)
        most_common_prediction = prediction_df["Prediction"].mode()[0]
        prediction_df.loc[
            prediction_df["Prediction"] != most_common_prediction,
            "Status",
        ] = "Different prediction"

        st.subheader("Model Prediction Cards")
        model_cols = st.columns(3)
        for index, row in prediction_df.iterrows():
            with model_cols[index]:
                st.metric(
                    label=row["Model"],
                    value=row["Prediction"],
                    delta=row["Top Score"],
                )
                st.caption(row["Status"])

        st.subheader("Comparison Table")
        st.dataframe(prediction_df, hide_index=True, use_container_width=True)
        st.caption(
            "Different prediction does not automatically mean wrong. "
            "For new user input, the real category is unknown unless manually checked."
        )


def render_evaluation_page():
    """
    Show benchmark evaluation metrics from training/testing.
    """
    st.title("Benchmark Evaluation Metrics")
    st.write(
        "This page shows model performance calculated from the testing split of the BBC News dataset. "
        "These values are not calculated from one user input."
    )

    comparison_df = load_model_comparison()
    if comparison_df is None:
        st.warning("Model comparison file not found. Please run the training scripts first.")
        return

    best_model = get_best_model(comparison_df)
    if best_model is not None:
        best_model_name, best_f1_score = best_model
        st.success(f"Best model: {best_model_name} with F1-score {(best_f1_score * 100):.2f}%")

    display_df = prepare_metrics_table(comparison_df)
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    st.caption("Accuracy, precision, recall, and F1-score are shown as percentages.")

    with st.expander("What do these metrics mean?"):
        st.write(
            """
            - Accuracy shows the overall percentage of correct predictions.
            - Precision shows how many predicted categories are actually correct.
            - Recall shows how many real category items are successfully detected.
            - F1-score balances precision and recall into one score.
            - CV F1-score comes from cross-validation during model tuning.
            """
        )


def render_about_page():
    """
    Explain the system workflow.
    """
    st.title("About NewsSort AI")
    st.write(
        """
        NewsSort AI is an NLP-based news category classification prototype.
        It classifies news text into five categories: business, entertainment, politics, sport, and tech.
        """
    )

    st.subheader("System Workflow")
    st.write(
        """
        1. Load the BBC News dataset.
        2. Use the `content` column as input text.
        3. Use the `category` column as the target label.
        4. Clean the news text by converting it to lowercase and removing unnecessary symbols.
        5. Convert the text into numerical features using word and character TF-IDF.
        6. Train Support Vector Machine, Logistic Regression, and Multinomial Naive Bayes.
        7. Compare the models using accuracy, precision, recall, and F1-score.
        8. Use the trained models inside the Streamlit prototype.
        """
    )

    st.subheader("Academic Note")
    st.info(
        "This system is for academic demonstration only. "
        "The prediction is based on patterns learned from the BBC News dataset."
    )


def main():
    st.set_page_config(
        page_title="NewsSort AI",
        page_icon="N",
        layout="wide",
    )
    apply_professional_news_blue_theme()

    page, selected_model_name = render_sidebar()

    if page == "Predict News Category":
        render_predict_page(selected_model_name)
    elif page == "Multi-Model Comparison":
        render_multi_model_page()
    elif page == "Model Evaluation Results":
        render_evaluation_page()
    else:
        render_about_page()


if __name__ == "__main__":
    main()
