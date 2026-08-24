import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from article_extraction import extract_article_from_url
from translation_utils import (
    detect_input_language,
    get_bilingual_category_label,
    get_translation_languages,
    prepare_text_for_prediction,
    translate_text,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SVM_MODEL_PATH = PROJECT_ROOT / "models" / "svm_model.pkl"

CATEGORIES = {
    "business": {
        "label": "Business",
        "description": "Company news, economy, finance, stock market, sales, profit, or industry growth.",
        "examples": "market growth, bank profit, company sales, investors, trade",
    },
    "entertainment": {
        "label": "Entertainment",
        "description": "Movies, music, celebrities, television, awards, festivals, or cultural events.",
        "examples": "film award, actor, singer, music festival, movie release",
    },
    "politics": {
        "label": "Politics",
        "description": "Government, elections, public policy, ministers, parliament, laws, or political leaders.",
        "examples": "election campaign, new policy, minister speech, parliament debate",
    },
    "sport": {
        "label": "Sport",
        "description": "Matches, teams, players, scores, tournaments, racing, coaching, or competitions.",
        "examples": "football match, final score, player goal, league tournament",
    },
    "tech": {
        "label": "Technology",
        "description": "Software, internet, artificial intelligence, devices, mobile phones, and digital innovation.",
        "examples": "AI feature, mobile app, software update, cloud platform",
    },
}

QUICK_SAMPLES = {
    "Tech": "Apple announced new artificial intelligence features for mobile devices, software developers, and cloud technology users.",
    "Business": "The company reported higher profit after strong sales growth, market recovery, and increased investor confidence.",
    "Politics": "The government announced a new policy after parliament debated election reform and public service funding.",
    "Sport": "The football team won the final match after the player scored a late goal in the tournament.",
    "Entertainment": "The actor received an award at the film festival after the movie became popular with audiences.",
}


def apply_black_purple_theme():
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #0B0B12;
                --sidebar-bg: #111827;
                --panel-bg: #151522;
                --panel-soft: #1E1B2E;
                --purple: #8B5CF6;
                --purple-dark: #7C3AED;
                --purple-soft: #C4B5FD;
                --text-main: #F8FAFC;
                --text-muted: #A1A1AA;
                --border: #2D2A3D;
                --success-bg: #12291F;
                --success-border: #22C55E;
            }

            .stApp {
                background: var(--app-bg);
                color: var(--text-main);
            }

            [data-testid="stHeader"] {
                background: var(--app-bg);
            }

            h1, h2, h3, h4, h5, h6, p, label, span, div {
                color: var(--text-main);
            }

            [data-testid="stSidebar"] {
                background: var(--sidebar-bg);
                border-right: 1px solid var(--border);
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] div {
                color: var(--text-main);
            }

            [data-testid="stSidebar"] .stCaptionContainer,
            [data-testid="stSidebar"] small,
            .stCaptionContainer {
                color: var(--text-muted);
            }

            textarea,
            input,
            [data-baseweb="select"] > div {
                background: var(--panel-soft) !important;
                color: var(--text-main) !important;
                border-color: var(--border) !important;
            }

            textarea::placeholder,
            input::placeholder {
                color: #71717A !important;
            }

            textarea:focus,
            input:focus {
                border-color: var(--purple) !important;
                box-shadow: 0 0 0 1px var(--purple) !important;
            }

            .stButton > button {
                border-radius: 8px;
                border: 1px solid var(--purple);
                background: var(--purple);
                color: #FFFFFF;
                font-weight: 700;
            }

            .stButton > button:hover {
                background: var(--purple-dark);
                border-color: var(--purple-dark);
                color: #FFFFFF;
            }

            [data-testid="stMetric"],
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--panel-bg);
                border: 1px solid var(--border);
                border-radius: 8px;
                box-shadow: none;
            }

            [data-testid="stMetricLabel"] p {
                color: var(--text-muted);
                font-weight: 700;
            }

            [data-testid="stMetricValue"] {
                color: var(--purple-soft);
            }

            [data-testid="stDataFrame"] {
                background: var(--panel-bg);
                border-radius: 8px;
            }

            .stAlert {
                border-radius: 8px;
            }

            hr {
                border-color: var(--border);
            }

            .category-card {
                background: var(--panel-bg);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 18px;
                min-height: 150px;
            }

            .result-card {
                background: var(--success-bg);
                border: 1px solid var(--success-border);
                border-radius: 8px;
                padding: 20px;
                margin-top: 18px;
            }

            .result-card h2 {
                margin: 0 0 8px 0;
                color: #DCFCE7;
            }

            .result-card p {
                margin-bottom: 0;
                color: #BBF7D0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_enough_news_text(news_text):
    text = str(news_text).strip()
    if not text:
        return False

    if len(text.split()) >= 10:
        return True

    compact_text = re.sub(r"\s+", "", text)
    return len(compact_text) >= 30


def get_text_stats(news_text):
    text = str(news_text).strip()
    words = re.findall(r"\b\w+\b", text)
    return len(words), len(text)


@st.cache_resource
def load_news_classifier():
    if not SVM_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {SVM_MODEL_PATH}")

    return joblib.load(SVM_MODEL_PATH)


def get_model_parts(model):
    return model.named_steps["features"], model.named_steps["model"]


def predict_with_details(news_text):
    model = load_news_classifier()
    cleaned_text = clean_text(news_text)
    prediction = model.predict([cleaned_text])[0]
    feature_extractor, classifier = get_model_parts(model)
    text_features = feature_extractor.transform([cleaned_text])

    decision_scores = classifier.decision_function(text_features)[0]
    score_df = build_category_match_table(classifier.classes_, decision_scores)
    key_terms = get_key_terms_for_prediction(
        feature_extractor,
        classifier,
        text_features,
        prediction,
    )

    return prediction, score_df, key_terms


def build_category_match_table(classes, decision_scores):
    scores = np.asarray(decision_scores, dtype=float)
    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        relative_scores = np.full_like(scores, 100 / len(scores), dtype=float)
    else:
        relative_scores = (scores - min_score) / (max_score - min_score) * 100

    score_df = pd.DataFrame(
        {
            "Category": [
                CATEGORIES.get(str(category), {}).get("label", str(category).title())
                for category in classes
            ],
            "Category Match": relative_scores.round(2),
        }
    ).sort_values("Category Match", ascending=False)

    return score_df


def get_key_terms_for_prediction(feature_extractor, classifier, text_features, prediction):
    try:
        feature_names = feature_extractor.get_feature_names_out()
        class_index = list(classifier.classes_).index(prediction)
        coefficients = classifier.coef_[class_index]
        contribution_values = text_features.multiply(coefficients).tocsr()[0]
    except (AttributeError, ValueError, IndexError):
        return []

    if contribution_values.nnz == 0:
        return []

    sorted_positions = contribution_values.data.argsort()[::-1]
    terms = []

    for position in sorted_positions:
        feature_index = contribution_values.indices[position]
        contribution = contribution_values.data[position]
        if contribution <= 0:
            continue

        term = (
            feature_names[feature_index]
            .replace("word_tfidf__", "")
            .replace("char_tfidf__", "")
            .strip()
        )
        if len(term) < 3 or term in terms:
            continue

        terms.append(term)
        if len(terms) == 8:
            break

    return terms


def get_prediction_strength(score_df):
    scores = score_df["Category Match"].tolist()
    if len(scores) < 2:
        return "Moderate"

    gap = scores[0] - scores[1]
    if gap >= 40:
        return "Strong"
    if gap >= 20:
        return "Moderate"
    return "Low"


def set_quick_sample(sample_name):
    st.session_state["news_text"] = QUICK_SAMPLES[sample_name]
    st.session_state["input_method"] = "Paste text"
    st.session_state.pop("article_url", None)
    st.session_state.pop("article_source", None)
    clear_prediction_result()


def clear_prediction_result():
    st.session_state.pop("prediction_result", None)


def clear_news_input():
    st.session_state["news_text"] = ""
    st.session_state["article_url"] = ""
    st.session_state.pop("article_source", None)
    clear_prediction_result()


def render_sidebar():
    st.sidebar.title("NewsSort AI")
    st.sidebar.caption("NLP News Category Classifier")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Classify News",
            "Category Guide",
            "About System",
        ],
    )

    st.sidebar.divider()
    st.sidebar.write("Language")
    translation_languages = list(get_translation_languages().keys())
    default_language_index = (
        translation_languages.index("English")
        if "English" in translation_languages
        else 0
    )
    translate_language = st.sidebar.selectbox(
        "Display Language",
        translation_languages,
        index=default_language_index,
    )
    show_translation = st.sidebar.checkbox(
        "Show translated article",
        value=False,
    )

    st.sidebar.divider()
    st.sidebar.write("Quick Test Samples")
    sample_cols = st.sidebar.columns(2)
    for index, sample_name in enumerate(QUICK_SAMPLES):
        with sample_cols[index % 2]:
            st.button(
                sample_name,
                use_container_width=True,
                on_click=set_quick_sample,
                args=(sample_name,),
            )

    st.sidebar.divider()
    st.sidebar.caption("Categories: Business, Entertainment, Politics, Sport, Technology")

    return page, translate_language, show_translation


def render_article_source_info():
    article_source = st.session_state.get("article_source")
    if not article_source:
        return

    st.success("Article text extracted successfully. You can edit the text below before classifying.")
    source_cols = st.columns(3)
    source_cols[0].metric("Extracted Words", article_source.get("word_count", 0))
    source_cols[1].metric("Source Domain", article_source.get("domain", "-"))
    source_cols[2].metric("Source Type", "URL")

    if article_source.get("title"):
        st.caption(f"Article title: {article_source['title']}")


def render_url_input():
    if "article_url" not in st.session_state:
        st.session_state["article_url"] = ""

    st.text_input(
        "News Article URL",
        key="article_url",
        placeholder="Paste a news article link here, for example https://www.bbc.com/news/...",
    )

    extract_clicked = st.button(
        "Extract Article Text",
        type="primary",
        use_container_width=True,
    )

    if extract_clicked:
        if not st.session_state["article_url"].strip():
            st.warning("Please paste a news article URL first.")
            return

        with st.spinner("Reading article from the link..."):
            extraction_result = extract_article_from_url(st.session_state["article_url"])

        if not extraction_result.success:
            st.error(extraction_result.error)
            st.info("If the website blocks extraction, paste the article text manually instead.")
            return

        st.session_state["news_text"] = extraction_result.text
        st.session_state["article_source"] = {
            "title": extraction_result.title,
            "domain": extraction_result.domain,
            "final_url": extraction_result.final_url,
            "word_count": extraction_result.word_count,
        }
        clear_prediction_result()


def render_news_input(input_method):
    if "news_text" not in st.session_state:
        st.session_state["news_text"] = ""

    if input_method == "Article URL":
        render_url_input()
        render_article_source_info()

    return st.text_area(
        "News Article Text",
        key="news_text",
        height=220,
        placeholder="Type or paste news text here, or use the URL option above to extract article text.",
    )


def render_translation_panel(original_text, translate_language):
    detected_language = detect_input_language(original_text)
    translated_display_text = translate_text(original_text, translate_language)

    with st.expander("Translated article", expanded=False):
        st.write(f"Detected language: {detected_language}")
        st.write(f"{translate_language} translation")
        st.write(translated_display_text)


def render_prediction_result(result, translate_language, show_translation):
    category = result["prediction"]
    category_label = get_bilingual_category_label(category, translate_language)
    category_info = CATEGORIES.get(str(category).lower(), {})
    strength = get_prediction_strength(result["score_df"])

    st.markdown(
        f"""
        <div class="result-card">
            <h2>Predicted Category: {category_label}</h2>
            <p>Prediction strength: {strength}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Why this category?")
    if category_info:
        st.write(category_info["description"])

    if result["key_terms"]:
        st.write("Key terms found in the article:")
        st.write(", ".join(result["key_terms"]))
    else:
        st.caption("No strong keyword signals were found, but the full text pattern was still used for classification.")

    st.subheader("Category Match")
    st.dataframe(
        result["score_df"],
        column_config={
            "Category Match": st.column_config.ProgressColumn(
                "Category Match",
                help="A relative match score for this input. It is not the model's training accuracy.",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            )
        },
        hide_index=True,
        use_container_width=True,
    )

    if show_translation:
        try:
            render_translation_panel(result["original_text"], translate_language)
        except RuntimeError as error:
            st.warning(str(error))

    st.caption(
        "Academic note: this prototype predicts a category based on patterns learned from the BBC News dataset."
    )


def render_classify_page(translate_language, show_translation):
    st.title("News Article Classifier")
    st.write(
        "Paste news text or extract it from an article link. The system will classify it as business, entertainment, politics, sport, or technology."
    )

    input_method = st.radio(
        "Input Method",
        ["Paste text", "Article URL"],
        key="input_method",
        horizontal=True,
    )

    news_text = render_news_input(input_method)
    word_count, character_count = get_text_stats(news_text)

    stat_cols = st.columns(2)
    stat_cols[0].metric("Word Count", word_count)
    stat_cols[1].metric("Character Count", character_count)

    button_cols = st.columns([1, 1])
    classify_clicked = button_cols[0].button(
        "Classify News",
        type="primary",
        use_container_width=True,
    )
    button_cols[1].button(
        "Clear Input",
        use_container_width=True,
        on_click=clear_news_input,
    )

    if classify_clicked:
        if not has_enough_news_text(news_text):
            st.warning("Please enter at least 10 words so the system has enough detail to classify.")
            return

        try:
            with st.spinner("Classifying news article..."):
                classifier_text = prepare_text_for_prediction(news_text)
                prediction, score_df, key_terms = predict_with_details(classifier_text)
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
            st.error(f"Prediction failed: {error}")
            return

        st.session_state["prediction_result"] = {
            "original_text": news_text,
            "classifier_text": classifier_text,
            "prediction": prediction,
            "score_df": score_df,
            "key_terms": key_terms,
        }

    result = st.session_state.get("prediction_result")
    if result and result["original_text"] == news_text:
        render_prediction_result(result, translate_language, show_translation)
    elif result:
        clear_prediction_result()


def render_category_guide_page():
    st.title("Category Guide")
    st.write("Use this guide to understand what each news category means.")

    rows = [
        ("business", "entertainment"),
        ("politics", "sport"),
        ("tech",),
    ]

    for row in rows:
        cols = st.columns(len(row))
        for column, category_key in zip(cols, row):
            category = CATEGORIES[category_key]
            with column:
                st.markdown(
                    f"""
                    <div class="category-card">
                        <h3>{category["label"]}</h3>
                        <p>{category["description"]}</p>
                        <p><strong>Typical words:</strong> {category["examples"]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.info(
        "For better results, paste at least one full paragraph instead of only one or two words."
    )


def render_about_page():
    st.title("About NewsSort AI")
    st.write(
        """
        NewsSort AI is an academic NLP prototype that classifies news articles into five categories:
        business, entertainment, politics, sport, and technology.
        """
    )

    st.subheader("How the system works")
    st.write(
        """
        1. The user pastes a news title or article.
        2. The system cleans the text and prepares it for classification.
        3. The trained classifier predicts the most suitable news category.
        4. The app shows the category, related key terms, and a relative category match table.
        """
    )

    st.subheader("Dataset and backend evaluation")
    st.write(
        """
        The project uses the BBC News dataset for training and testing. During backend development,
        several classification approaches were trained and compared using accuracy, precision, recall,
        and F1-score. The best-performing approach is used in this user-facing prototype.
        """
    )

    st.subheader("Limitation")
    st.write(
        """
        The system is trained on BBC-style news data, so it may be less accurate for very short text,
        mixed-topic articles, informal social media posts, or news topics outside the training categories.
        """
    )


def main():
    st.set_page_config(
        page_title="NewsSort AI",
        page_icon="N",
        layout="wide",
    )
    apply_black_purple_theme()

    page, translate_language, show_translation = render_sidebar()

    if page == "Classify News":
        render_classify_page(translate_language, show_translation)
    elif page == "Category Guide":
        render_category_guide_page()
    else:
        render_about_page()


if __name__ == "__main__":
    main()
