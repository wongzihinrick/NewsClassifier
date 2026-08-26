import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from article_extraction import extract_article_from_url
from news_features import (
    generate_core_takeaways,
    render_tts_player,
)
from translation_utils import (
    TRANSLATION_LANGUAGE_OPTIONS,
    detect_input_language,
    get_bilingual_category_label,
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
    "Tech": (
        "Artificial intelligence firm OpenAI has revealed its flagship model, GPT-4o, capable of realistic voice "
        "conversations and real-time interaction across text, vision, and audio. The company demonstrated the system "
        "translating spoken foreign languages in real time, detecting human facial expressions, and solving complex "
        "mathematics problems directly from camera feeds.\n\n"
        "Chief executive Sam Altman stated that the update represents a major technological leap, offering faster "
        "processing speeds and reduced latency across mobile apps and desktop software. Industry analysts noted that "
        "making these advanced multimodal capabilities accessible to all users for free significantly escalates competition "
        "in the global artificial intelligence and cloud computing market."
    ),
    "Business": (
        "The Bank of England has reduced its benchmark interest rate to 5% from 5.25%, marking the first cut in "
        "borrowing costs in more than four years. Policymakers on the Monetary Policy Committee voted by a narrow 5-4 "
        "majority in favour of the reduction, following evidence of cooling domestic price pressures and headline "
        "inflation falling to the official 2% target.\n\n"
        "The move provides relief for mortgage holders and corporate borrowers after two years of steep interest rate "
        "increases designed to combat soaring consumer prices. However, Governor Andrew Bailey emphasised that the central "
        "bank will remain cautious about future reductions, warning that persistent wage growth and services inflation must "
        "continue to ease before further monetary easing can take place."
    ),
    "Politics": (
        "Sir Keir Starmer has promised to lead a 'government of service' on an urgent mission of national renewal "
        "following Labour's landslide general election victory that ended 14 years of Conservative administration. Delivering "
        "his first address outside 10 Downing Street after formally accepting the King's invitation to form a government, "
        "the new Prime Minister stated that the public had delivered a decisive verdict for institutional reform.\n\n"
        "He pledged to restore public trust in government, rebuild crumbling public infrastructure, and prioritize economic "
        "stability over party political interests. In the coming weeks, the new cabinet ministers will outline legislative "
        "priorities in the King's Speech, focusing heavily on NHS funding, regional housing expansion, and renewable energy "
        "investments across the United Kingdom."
    ),
    "Sport": (
        "Spain's Carlos Alcaraz produced a masterclass performance on Centre Court to sweep past seven-time champion "
        "Novak Djokovic in straight sets and retain his Wimbledon gentlemen's singles title. The 21-year-old Spaniard "
        "dominated from the baseline with overwhelming power and delicate touch, weathering a brief comeback attempt in the "
        "third set before closing out a 6-2 6-2 7-6 triumph.\n\n"
        "The victory sealed the fourth Grand Slam crown of Alcaraz's young career, having already won the French Open just a "
        "month earlier. Speaking during the trophy presentation, Alcaraz praised Djokovic as an inspiration to the sport "
        "and expressed immense pride in defending his title in front of royalty and a capacity crowd at the All England Club."
    ),
    "Entertainment": (
        "Christopher Nolan's biographical epic Oppenheimer dominated the 96th Academy Awards in Hollywood, securing seven "
        "Oscars including Best Picture, Best Director, and Best Actor for Irish star Cillian Murphy. The three-hour historical "
        "drama chronicling the life of theoretical physicist J. Robert Oppenheimer and the Manhattan Project entered the "
        "ceremony with 13 nominations and swept major competitive categories.\n\n"
        "Robert Downey Jr collected the Best Supporting Actor accolade, while the film also claimed trophies for best film "
        "editing, cinematography, and original score. In his acceptance speech, Nolan thanked universal audiences for "
        "supporting cinema on the big screen, celebrating the collaborative vision of the cast and crew behind the blockbuster release."
    ),
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

            .takeaways-card {
                background: linear-gradient(135deg, #181428 0%, #151522 100%);
                border: 1px solid #3F3366;
                border-left: 4px solid var(--purple);
                border-radius: 8px;
                padding: 16px 20px;
                margin: 16px 0;
            }

            .takeaways-card h4 {
                margin: 0 0 10px 0;
                color: var(--purple-soft);
                font-size: 16px;
                font-weight: 700;
            }

            .takeaway-point {
                background: #1F1A33;
                border: 1px solid #362D55;
                border-radius: 6px;
                padding: 10px 14px;
                margin-bottom: 8px;
                color: #E2E8F0;
                font-size: 14px;
                line-height: 1.5;
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


def render_live_text_stats(initial_text):
    word_count, character_count = get_text_stats(initial_text)
    components.html(
        f"""
        <style>
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 16px;
                font-family: "Source Sans Pro", sans-serif;
            }}

            .stat-card {{
                background: #151522;
                border: 1px solid #2D2A3D;
                border-radius: 8px;
                padding: 10px 14px 14px;
                min-height: 78px;
                box-sizing: border-box;
            }}

            .stat-label {{
                color: #A1A1AA;
                font-size: 14px;
                font-weight: 700;
                margin-bottom: 8px;
            }}

            .stat-value {{
                color: #F8FAFC;
                font-size: 34px;
                line-height: 1;
                font-weight: 500;
            }}
        </style>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Word Count</div>
                <div class="stat-value" id="live-word-count">{word_count}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Character Count</div>
                <div class="stat-value" id="live-character-count">{character_count}</div>
            </div>
        </div>
        <script>
            const wordCount = document.getElementById("live-word-count");
            const characterCount = document.getElementById("live-character-count");

            function countWords(text) {{
                const trimmed = text.trim();
                if (!trimmed) return 0;
                return Array.from(
                    trimmed.matchAll(/[\\p{{L}}\\p{{N}}]+/gu)
                ).length;
            }}

            function updateCounts() {{
                const textArea = Array.from(window.parent.document.querySelectorAll("textarea"))
                    .find((element) => element.getAttribute("aria-label") === "News Article Text");

                if (!textArea) return;

                wordCount.textContent = countWords(textArea.value);
                characterCount.textContent = textArea.value.trim().length;
            }}

            updateCounts();
            const intervalId = window.setInterval(updateCounts, 250);
            window.addEventListener("beforeunload", () => window.clearInterval(intervalId));
        </script>
        """,
        height=116,
    )


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
    clear_article_source()
    clear_prediction_result()


def clear_prediction_result():
    st.session_state.pop("prediction_result", None)


def clear_article_source():
    st.session_state.pop("article_source", None)


def clear_news_input():
    st.session_state["news_text"] = ""
    st.session_state["article_url"] = ""
    clear_article_source()
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
    translate_language = st.sidebar.selectbox(
        "Translate Language",
        TRANSLATION_LANGUAGE_OPTIONS,
        index=0,
    )
    show_translation = st.sidebar.checkbox(
        "Show translated article",
        value=True,
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

        clear_article_source()
        clear_prediction_result()

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

    with st.expander("Translated article", expanded=False):
        st.write(f"Detected language: {detected_language}")
        st.write(f"{translate_language} translation")
        with st.spinner(f"Translating full article to {translate_language}..."):
            translated_display_text = translate_text(original_text, translate_language)
        st.write(translated_display_text)


def render_prediction_result(result, translate_language, show_translation):
    category = result["prediction"]
    category_label = get_bilingual_category_label(category, translate_language)
    category_info = CATEGORIES.get(str(category).lower(), {})
    strength = get_prediction_strength(result["score_df"])
    takeaways = result.get("takeaways", [])

    st.markdown(
        f"""
        <div class="result-card">
            <h2>Predicted Category: {category_label}</h2>
            <p>Prediction strength: {strength}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if takeaways:
        st.markdown(
            """
            <div class="takeaways-card">
                <h4>📝 3-Sentence Core Takeaways</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for point in takeaways:
            st.markdown(
                f'<div class="takeaway-point">{point}</div>',
                unsafe_allow_html=True,
            )

    tts_content = " ".join(takeaways) if takeaways else result["original_text"][:250]
    render_tts_player(tts_content, label="🔊 Listen to 3-Sentence Core Takeaways")

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
    render_live_text_stats(news_text)

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
                takeaways = generate_core_takeaways(news_text, max_points=3)
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
            st.error(f"Prediction failed: {error}")
            return

        pred_result = {
            "original_text": news_text,
            "classifier_text": classifier_text,
            "prediction": prediction,
            "score_df": score_df,
            "key_terms": key_terms,
            "takeaways": takeaways,
        }
        st.session_state["prediction_result"] = pred_result

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
