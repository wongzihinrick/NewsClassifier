import io
import math
import re
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import streamlit.components.v1 as components


# ==========================================
# 1. 3-Sentence Core Takeaways Summarizer
# ==========================================

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "said", "also", "would", "one", "two", "new", "mr",
    "year", "years", "people", "first", "last"
}


def split_into_sentences(text):
    """Split text into sentences cleanly."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def generate_core_takeaways(text, max_points=3):
    """
    Extract 3 core takeaways with distinct emojis (Event, Impact/Detail, Context/Next).
    Uses sentence position, length, and keyword significance scoring.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [f"📌 {text[:120]}..."]

    if len(sentences) <= max_points:
        emojis = ["📌", "💡", "🔍"]
        return [f"{emojis[i % len(emojis)]} {s}" for i, s in enumerate(sentences)]

    # Compute word frequencies
    words = re.findall(r"\b[a-zA-Z\u4e00-\u9fa5]{3,}\b", text.lower())
    freq = {}
    for w in words:
        if w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1

    max_f = max(freq.values()) if freq else 1
    for w in freq:
        freq[w] /= max_f

    # Score sentences
    scored_sentences = []
    for idx, sentence in enumerate(sentences):
        s_words = re.findall(r"\b[a-zA-Z\u4e00-\u9fa5]{3,}\b", sentence.lower())
        if not s_words:
            continue

        score = sum(freq.get(w, 0) for w in s_words) / (len(s_words) ** 0.6)

        # Boost lead sentence (core event)
        if idx == 0:
            score *= 1.5
        elif idx == 1:
            score *= 1.2
        elif idx == len(sentences) - 1:
            score *= 1.1

        scored_sentences.append((score, idx, sentence))

    # Pick top N sentences maintaining chronological order
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_picks = sorted(scored_sentences[:max_points], key=lambda x: x[1])

    emojis = ["📌", "💡", "🔍"]
    takeaways = []
    for i, (_, _, sentence) in enumerate(top_picks):
        takeaways.append(f"{emojis[i % len(emojis)]} {sentence}")

    return takeaways


# ==========================================
# 2. Text-to-Speech (TTS) Web Component
# ==========================================

def render_tts_player(text_to_speak, label="🔊 Listen to 3-Sentence Core Takeaways"):
    """
    Render a zero-latency Web Speech API audio player inside Streamlit.
    Works client-side directly in the browser across Chrome, Edge, Safari, Firefox.
    """
    # Sanitize text for JavaScript string literal
    safe_text = (
        text_to_speak.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("\n", " ")
        .replace('"', '\\"')
    )

    component_html = f"""
    <div style="
        background: #181528;
        border: 1px solid #3A3258;
        border-radius: 10px;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="
                width: 36px;
                height: 36px;
                background: linear-gradient(135deg, #8B5CF6, #6D28D9);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
            ">🎙️</div>
            <div>
                <div style="font-size: 14px; font-weight: 600; color: #F8FAFC;">{label}</div>
                <div id="tts-status" style="font-size: 11px; color: #94A3B8;">Click play to listen</div>
            </div>
        </div>
        <div style="display: flex; gap: 8px;">
            <button id="tts-play-btn" onclick="startSpeech()" style="
                background: #8B5CF6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 13px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
            ">▶ Play</button>
            <button id="tts-stop-btn" onclick="stopSpeech()" style="
                background: #332D48;
                color: #CBD5E1;
                border: 1px solid #4C4468;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
                cursor: pointer;
            ">⏹ Stop</button>
        </div>
    </div>

    <script>
        const contentToRead = "{safe_text}";
        let utterance = null;

        function startSpeech() {{
            if (!('speechSynthesis' in window)) {{
                document.getElementById('tts-status').innerText = "Speech synthesis not supported in this browser";
                return;
            }}
            window.speechSynthesis.cancel();
            utterance = new SpeechSynthesisUtterance(contentToRead);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;

            utterance.onstart = function() {{
                document.getElementById('tts-status').innerText = "Playing audio...";
                document.getElementById('tts-play-btn').innerText = "⏸ Pause";
                document.getElementById('tts-play-btn').onclick = pauseSpeech;
            }};

            utterance.onend = function() {{
                resetTTS();
            }};

            utterance.onerror = function() {{
                resetTTS();
            }};

            window.speechSynthesis.speak(utterance);
        }}

        function pauseSpeech() {{
            if (window.speechSynthesis.speaking) {{
                if (window.speechSynthesis.paused) {{
                    window.speechSynthesis.resume();
                    document.getElementById('tts-status').innerText = "Playing audio...";
                    document.getElementById('tts-play-btn').innerText = "⏸ Pause";
                }} else {{
                    window.speechSynthesis.pause();
                    document.getElementById('tts-status').innerText = "Paused";
                    document.getElementById('tts-play-btn').innerText = "▶ Resume";
                }}
            }}
        }}

        function stopSpeech() {{
            window.speechSynthesis.cancel();
            resetTTS();
        }}

        function resetTTS() {{
            document.getElementById('tts-status').innerText = "Playback finished";
            document.getElementById('tts-play-btn').innerText = "▶ Play";
            document.getElementById('tts-play-btn').onclick = startSpeech;
        }}
    </script>
    """
    components.html(component_html, height=72)


# ==========================================
# 3. Shareable Visual News Card Generator
# ==========================================

def detect_script(text):
    """Detect if text contains Tamil, Arabic, CJK, or Latin characters."""
    if not text:
        return "latin"
    if re.search(r"[\u0b80-\u0bff]", text):
        return "tamil"
    if re.search(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]", text):
        return "arabic"
    if re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", text):
        return "cjk"
    return "latin"


def get_unicode_font(size, bold=False, sample_text=""):
    """
    Load the appropriate font supporting English, Chinese (CJK), Tamil (Indic), or Arabic.
    """
    script = detect_script(sample_text)
    font_candidates = []

    if script == "tamil":
        font_candidates = [
            "C:/Windows/Fonts/Nirmala.ttc",
            "Nirmala.ttc",
            "C:/Windows/Fonts/latha.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
    elif script == "arabic":
        font_candidates = [
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/tahomabd.ttf" if bold else "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "segoeui.ttf",
            "tahoma.ttf",
        ]
    elif script == "cjk":
        font_candidates = [
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "msyh.ttc",
            "simsun.ttc",
        ]
    else:
        font_candidates = [
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        ]

    # Universal fallbacks
    font_candidates += [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/Nirmala.ttc",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def wrap_text(text, font, max_width, draw):
    """Wrap text to fit within a maximum pixel width, supporting Latin, Arabic, Tamil, and CJK text."""
    if not text:
        return []

    # Check if text contains spaces or is mostly character tokens
    has_spaces = " " in text.strip()
    tokens = text.split(" ") if has_spaces else list(text)
    delimiter = " " if has_spaces else ""

    lines = []
    current_line = []

    for token in tokens:
        test_line = delimiter.join(current_line + [token])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]
        if line_width <= max_width:
            current_line.append(token)
        else:
            if current_line:
                lines.append(delimiter.join(current_line))
            current_line = [token]

    if current_line:
        lines.append(delimiter.join(current_line))
    return lines


def generate_news_card_image(
    title,
    category_name,
    strength,
    takeaways,
    key_terms=None,
    source_domain="News Article",
):
    """
    Generate an elegant, high-resolution dark-mode social share card (PNG image).
    """
    width = 900
    height = 680

    # Create background image with purple/dark gradient feel
    image = Image.new("RGB", (width, height), color="#0F0C1B")
    draw = ImageDraw.Draw(image)

    # Draw gradient-like background panels and neon borders
    draw.rectangle([(20, 20), (width - 20, height - 20)], fill="#171427", outline="#2F2848", width=2)
    draw.rectangle([(20, 20), (width - 20, 100)], fill="#211C38")

    # Load Unicode-compatible fonts dynamically based on language / script
    cat_text = f"CATEGORY: {category_name.upper()}"
    clean_title = title if title else "News Analysis & Core Takeaways"
    takeaways_sample = " ".join(takeaways) if takeaways else ""

    font_badge = get_unicode_font(18, bold=True, sample_text=cat_text)
    font_title = get_unicode_font(24, bold=True, sample_text=clean_title)
    font_header = get_unicode_font(16, bold=True)
    font_body = get_unicode_font(15, bold=False, sample_text=takeaways_sample)
    font_small = get_unicode_font(13, bold=False)

    # Header: Brand & Domain
    draw.text((45, 38), "NEWSSORT AI", fill="#8B5CF6", font=font_badge)
    draw.text((45, 68), f"Source: {source_domain}", fill="#94A3B8", font=font_small)

    # Category Badge on Top Right
    cat_bbox = draw.textbbox((0, 0), cat_text, font=font_badge)
    cat_w = cat_bbox[2] - cat_bbox[0] + 30
    badge_x = width - 45 - cat_w
    draw.rounded_rectangle([(badge_x, 38), (width - 45, 82)], radius=8, fill="#6D28D9")
    draw.text((badge_x + 15, 48), cat_text, fill="#FFFFFF", font=font_badge)

    # Article Title / Main Topic
    curr_y = 125
    clean_title = title if title else "News Analysis & Core Takeaways"
    title_lines = wrap_text(clean_title, font_title, width - 90, draw)
    for line in title_lines[:2]:
        draw.text((45, curr_y), line, fill="#F8FAFC", font=font_title)
        curr_y += 34

    # Sub-header bar
    curr_y += 10
    draw.line([(45, curr_y), (width - 45, curr_y)], fill="#362F52", width=1)
    curr_y += 20

    # Section Title: Core Takeaways
    draw.text((45, curr_y), "KEY TAKEAWAYS (3-POINT SUMMARY)", fill="#A78BFA", font=font_header)
    curr_y += 32

    # Draw Takeaway Items in Cards
    for item in takeaways[:3]:
        card_start_y = curr_y
        # Clean item for PIL drawing (replace emoji with ascii tag for safe font rendering)
        clean_item = item.replace("📌", "[1]").replace("💡", "[2]").replace("🔍", "[3]").strip()
        lines = wrap_text(clean_item, font_body, width - 130, draw)
        item_height = max(44, len(lines) * 24 + 16)

        draw.rounded_rectangle(
            [(45, card_start_y), (width - 45, card_start_y + item_height)],
            radius=6,
            fill="#1E1933",
            outline="#352C55",
            width=1,
        )

        text_y = card_start_y + 10
        for l in lines:
            draw.text((65, text_y), l, fill="#E2E8F0", font=font_body)
            text_y += 22

        curr_y += item_height + 12

    # Section: Key Terms & Strength
    curr_y += 10
    if key_terms:
        draw.text((45, curr_y), "KEY SIGNAL TERMS:", fill="#94A3B8", font=font_small)
        terms_str = "  *  ".join(key_terms[:6])
        draw.text((195, curr_y), terms_str, fill="#C4B5FD", font=font_small)
        curr_y += 26

    draw.text((45, curr_y), f"Classification Confidence / Strength: {strength}", fill="#94A3B8", font=font_small)

    # Footer
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    draw.line([(45, height - 60), (width - 45, height - 60)], fill="#2E2749", width=1)
    draw.text((45, height - 45), f"Generated by NewsSort AI  |  {time_str}", fill="#64748B", font=font_small)
    draw.text((width - 240, height - 45), "NLP Automated Classification", fill="#64748B", font=font_small)

    # Return bytes buffer
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf


