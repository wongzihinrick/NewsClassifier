import re

import streamlit as st


CATEGORY_TRANSLATIONS = {
    "business": {
        "English": "Business",
        "Arabic": "الأعمال",
        "Chinese": "商业",
        "French": "Affaires",
        "German": "Wirtschaft",
        "Hindi": "व्यापार",
        "Indonesian": "Bisnis",
        "Japanese": "ビジネス",
        "Korean": "비즈니스",
        "Malay": "Perniagaan",
        "Portuguese": "Negócios",
        "Russian": "Бизнес",
        "Spanish": "Negocios",
        "Tamil": "வணிகம்",
    },
    "entertainment": {
        "English": "Entertainment",
        "Arabic": "الترفيه",
        "Chinese": "娱乐",
        "French": "Divertissement",
        "German": "Unterhaltung",
        "Hindi": "मनोरंजन",
        "Indonesian": "Hiburan",
        "Japanese": "エンタメ",
        "Korean": "엔터테인먼트",
        "Malay": "Hiburan",
        "Portuguese": "Entretenimento",
        "Russian": "Развлечения",
        "Spanish": "Entretenimiento",
        "Tamil": "பொழுதுபோக்கு",
    },
    "politics": {
        "English": "Politics",
        "Arabic": "السياسة",
        "Chinese": "政治",
        "French": "Politique",
        "German": "Politik",
        "Hindi": "राजनीति",
        "Indonesian": "Politik",
        "Japanese": "政治",
        "Korean": "정치",
        "Malay": "Politik",
        "Portuguese": "Política",
        "Russian": "Политика",
        "Spanish": "Política",
        "Tamil": "அரசியல்",
    },
    "sport": {
        "English": "Sport",
        "Arabic": "الرياضة",
        "Chinese": "体育",
        "French": "Sport",
        "German": "Sport",
        "Hindi": "खेल",
        "Indonesian": "Olahraga",
        "Japanese": "スポーツ",
        "Korean": "스포츠",
        "Malay": "Sukan",
        "Portuguese": "Esporte",
        "Russian": "Спорт",
        "Spanish": "Deportes",
        "Tamil": "விளையாட்டு",
    },
    "tech": {
        "English": "Tech",
        "Arabic": "التقنية",
        "Chinese": "科技",
        "French": "Technologie",
        "German": "Technologie",
        "Hindi": "तकनीक",
        "Indonesian": "Teknologi",
        "Japanese": "テクノロジー",
        "Korean": "기술",
        "Malay": "Teknologi",
        "Portuguese": "Tecnologia",
        "Russian": "Технологии",
        "Spanish": "Tecnología",
        "Tamil": "தொழில்நுட்பம்",
    },
}

FALLBACK_LANGUAGE_CODES = {
    "Afrikaans": "af",
    "Arabic": "ar",
    "Bengali": "bn",
    "Chinese": "zh-CN",
    "Dutch": "nl",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Malay": "ms",
    "Portuguese": "pt",
    "Russian": "ru",
    "Spanish": "es",
    "Tamil": "ta",
    "Thai": "th",
    "Vietnamese": "vi",
}

DETECTED_LANGUAGE_NAMES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "ms": "Malay",
    "ta": "Tamil",
}

MYMEMORY_LANGUAGE_CODES = {
    "ar": "ar-SA",
    "de": "de-DE",
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "hi": "hi-IN",
    "id": "id-ID",
    "it": "it-IT",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "ms": "ms-MY",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "ta": "ta-IN",
    "th": "th-TH",
    "vi": "vi-VN",
    "zh-CN": "zh-CN",
    "zh-TW": "zh-TW",
}


def get_category_label(category, language):
    """
    Return a category label in the selected display language.
    """
    category_key = str(category).lower()
    english_label = CATEGORY_TRANSLATIONS.get(category_key, {}).get(
        "English",
        str(category).title(),
    )
    manual_label = CATEGORY_TRANSLATIONS.get(category_key, {}).get(language)

    if language == "English":
        return english_label

    if manual_label:
        return manual_label

    return translate_category_label(english_label, language)


@st.cache_data(show_spinner=False)
def translate_category_label(english_label, target_language):
    """
    Translate a short category label without allowing translator errors to break the app.
    """
    if target_language == "English":
        return english_label

    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return english_label

    target_code = get_language_code(target_language)
    translated_label = try_google_translate(english_label, target_code, ["auto", "en"])

    if not translated_label:
        return english_label

    return translated_label


@st.cache_data(show_spinner=False)
def get_translation_languages():
    """
    Return all target languages supported by Google Translate.
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return FALLBACK_LANGUAGE_CODES

    supported_languages = GoogleTranslator().get_supported_languages(as_dict=True)
    language_codes = {
        language_name.title(): language_code
        for language_name, language_code in supported_languages.items()
    }

    if "Chinese (Simplified)" in language_codes:
        language_codes["Chinese"] = language_codes["Chinese (Simplified)"]

    return dict(sorted(language_codes.items()))


def get_language_code(language):
    """
    Return the translation code for a display language.
    """
    return get_translation_languages().get(language, FALLBACK_LANGUAGE_CODES.get(language, "en"))


def try_google_translate(text, target_code, source_codes):
    """
    Try Google Translate first, then MyMemory for common language pairs.
    """
    from deep_translator import GoogleTranslator

    for source_code in source_codes:
        try:
            translated_text = GoogleTranslator(
                source=source_code,
                target=target_code,
            ).translate(text)
        except Exception:
            continue

        if translated_text and not is_bad_translation_result(translated_text):
            return translated_text

    mymemory_target_code = MYMEMORY_LANGUAGE_CODES.get(target_code)
    if not mymemory_target_code:
        return None

    try:
        from deep_translator import MyMemoryTranslator
    except ImportError:
        return None

    for source_code in source_codes:
        if source_code == "auto":
            continue

        mymemory_source_code = MYMEMORY_LANGUAGE_CODES.get(source_code)
        if not mymemory_source_code:
            continue

        try:
            translated_text = MyMemoryTranslator(
                source=mymemory_source_code,
                target=mymemory_target_code,
            ).translate(text)
        except Exception:
            continue

        if translated_text and not is_bad_translation_result(translated_text):
            return translated_text

    return None


def is_bad_translation_result(translated_text):
    """
    Detect error pages returned as text by unofficial translation endpoints.
    """
    normalized_text = str(translated_text).lower()
    error_signals = [
        "error 500",
        "server error",
        "that's an error",
        "there was an error",
        "no translation was found",
    ]
    return any(signal in normalized_text for signal in error_signals)


def get_bilingual_category_label(category, language):
    """
    Return an English category label plus the selected translated label.
    """
    english_label = get_category_label(category, "English")
    translated_label = get_category_label(category, language)

    if language == "English" or translated_label == english_label:
        return english_label

    return f"{english_label} / {translated_label}"


def detect_language_by_script(text):
    """
    Detect languages with distinctive writing systems before statistical detection.
    """
    lower_text = text.lower()
    malay_markers = [
        "baharu",
        "kerajaan",
        "parlimen",
        "perkhidmatan",
        "awam",
        "sukan",
    ]
    if any(marker in lower_text for marker in malay_markers):
        return "Malay"

    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese"

    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean"

    if re.search(r"[\u0600-\u06ff]", text):
        return "Arabic"

    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"

    if re.search(r"[\u0b80-\u0bff]", text):
        return "Tamil"

    return None


@st.cache_data(show_spinner=False)
def detect_input_language(text):
    """
    Detect the input language for smarter translation decisions.
    """
    cleaned_text = str(text).strip()
    if not cleaned_text:
        return "Unknown"

    script_language = detect_language_by_script(cleaned_text)
    if script_language:
        return script_language

    try:
        from langdetect import DetectorFactory, LangDetectException, detect
    except ImportError:
        return "Unknown"

    DetectorFactory.seed = 0

    try:
        detected_code = detect(cleaned_text).lower()
    except LangDetectException:
        return "Unknown"

    return DETECTED_LANGUAGE_NAMES.get(detected_code, "Unknown")


@st.cache_data(show_spinner=False)
def translate_text(text, target_language):
    """
    Auto-detect the input language and translate to the target language.
    """
    if not text.strip():
        return text

    detected_language = detect_input_language(text)
    if detected_language == target_language:
        return text

    try:
        from deep_translator import GoogleTranslator
    except ImportError as error:
        raise RuntimeError(
            "Translation package is missing. Run: pip install -r requirements.txt"
        ) from error

    target_code = get_language_code(target_language)
    source_codes = ["auto"]
    if detected_language != "Unknown":
        detected_source_code = get_language_code(detected_language)
        if detected_source_code not in source_codes:
            source_codes.append(detected_source_code)

    translated_text = try_google_translate(text, target_code, source_codes)

    if not translated_text:
        raise RuntimeError(
            f"Translation to {target_language} failed. Please try another language or try again later."
        )

    return translated_text


def prepare_text_for_prediction(news_text):
    """
    Auto-detect and translate input into English before classification.
    """
    if detect_input_language(news_text) == "English":
        return news_text

    return translate_text(news_text, "English")
