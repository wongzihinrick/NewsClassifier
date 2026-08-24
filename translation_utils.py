import re

import streamlit as st


CATEGORY_TRANSLATIONS = {
    "business": {
        "English": "Business",
        "Arabic": "الأعمال",
        "Chinese (Simplified)": "商业",
        "Chinese (Traditional)": "商業",
        "Malay": "Perniagaan",
        "Tamil": "வணிகம்",
    },
    "entertainment": {
        "English": "Entertainment",
        "Arabic": "الترفيه",
        "Chinese (Simplified)": "娱乐",
        "Chinese (Traditional)": "娛樂",
        "Malay": "Hiburan",
        "Tamil": "பொழுதுபோக்கு",
    },
    "politics": {
        "English": "Politics",
        "Arabic": "السياسة",
        "Chinese (Simplified)": "政治",
        "Chinese (Traditional)": "政治",
        "Malay": "Politik",
        "Tamil": "அரசியல்",
    },
    "sport": {
        "English": "Sport",
        "Arabic": "الرياضة",
        "Chinese (Simplified)": "体育",
        "Chinese (Traditional)": "體育",
        "Malay": "Sukan",
        "Tamil": "விளையாட்டு",
    },
    "tech": {
        "English": "Tech",
        "Arabic": "التقنية",
        "Chinese (Simplified)": "科技",
        "Chinese (Traditional)": "科技",
        "Malay": "Teknologi",
        "Tamil": "தொழில்நுட்பம்",
    },
}

TRANSLATION_LANGUAGE_OPTIONS = [
    "English",
    "Malay",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
    "Tamil",
    "Arabic",
]

FALLBACK_LANGUAGE_CODES = {
    "Arabic": "ar",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "English": "en",
    "Malay": "ms",
    "Tamil": "ta",
}

DETECTED_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "en": "English",
    "zh": "Chinese (Simplified)",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ms": "Malay",
    "ta": "Tamil",
}

MYMEMORY_LANGUAGE_CODES = {
    "ar": "ar-SA",
    "en": "en-US",
    "ms": "ms-MY",
    "ta": "ta-IN",
    "zh-CN": "zh-CN",
    "zh-TW": "zh-TW",
}
MAX_TRANSLATION_CHARS = 1000


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
    translated_label = try_google_translate(
        english_label,
        target_code,
        ["auto", "en"],
        source_language="English",
        target_language=target_language,
    )

    if not translated_label:
        return english_label

    return translated_label


@st.cache_data(show_spinner=False)
def get_translation_languages():
    """
    Return only the target languages supported by this app.
    """
    return FALLBACK_LANGUAGE_CODES


def get_language_code(language):
    """
    Return the translation code for a display language.
    """
    return get_translation_languages().get(language, FALLBACK_LANGUAGE_CODES.get(language, "en"))


def try_google_translate(
    text,
    target_code,
    source_codes,
    source_language=None,
    target_language=None,
):
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

        if (
            translated_text
            and not is_bad_translation_result(translated_text)
            and not has_untranslated_source_text(text, translated_text, source_language, target_language)
        ):
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

        if (
            translated_text
            and not is_bad_translation_result(translated_text)
            and not has_untranslated_source_text(text, translated_text, source_language, target_language)
        ):
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


def count_cjk_chars(text):
    """
    Count Chinese/Japanese/Korean unified ideograph characters.
    """
    return len(re.findall(r"[\u4e00-\u9fff]", str(text)))


def has_untranslated_source_text(source_text, translated_text, source_language, target_language):
    """
    Reject partial translations that still contain too much source script.
    """
    if not str(source_language).startswith("Chinese") or str(target_language).startswith("Chinese"):
        return False

    source_cjk_count = count_cjk_chars(source_text)
    translated_cjk_count = count_cjk_chars(translated_text)

    if source_cjk_count < 10:
        return False

    return translated_cjk_count > max(6, int(source_cjk_count * 0.08))


def split_text_for_translation(text, max_chars=MAX_TRANSLATION_CHARS):
    """
    Split long article text by paragraph and sentence for more accurate translation.
    """
    normalized_text = str(text).strip()
    if len(normalized_text) <= max_chars:
        return [normalized_text]

    chunks = []
    current_chunk = ""
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", normalized_text)]

    for paragraph in paragraphs:
        if not paragraph:
            continue

        sentences = split_paragraph_for_translation(paragraph, max_chars)

        for sentence in sentences:
            if not sentence:
                continue

            candidate = sentence if not current_chunk else f"{current_chunk}\n\n{sentence}"
            if len(candidate) <= max_chars:
                current_chunk = candidate
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def split_paragraph_for_translation(paragraph, max_chars):
    """
    Split one paragraph without breaking sentences unless the sentence is too long.
    """
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?。！？])\s*", paragraph)
    pieces = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue

        for index in range(0, len(sentence), max_chars):
            pieces.append(sentence[index:index + max_chars].strip())

    return pieces


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

    if re.search(r"[\u0600-\u06ff]", text):
        return "Arabic"

    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese (Simplified)"

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

    translated_chunks = []
    for chunk in split_text_for_translation(text):
        translated_chunk = try_google_translate(
            chunk,
            target_code,
            source_codes,
            source_language=detected_language,
            target_language=target_language,
        )

        if not translated_chunk:
            raise RuntimeError(
                f"Translation to {target_language} failed before the full article was translated. "
                "Please try again later."
            )

        translated_chunks.append(translated_chunk)

    return "\n\n".join(translated_chunks)


def prepare_text_for_prediction(news_text):
    """
    Auto-detect and translate input into English before classification.
    """
    if detect_input_language(news_text) == "English":
        return news_text

    return translate_text(news_text, "English")
