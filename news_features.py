import re
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



