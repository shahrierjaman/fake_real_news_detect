# ============================================================
#  Fake News Detector — Streamlit App  (app.py)
#  Run: streamlit run app.py
#  Requires: fake_news_bert/  fake_news_metrics.json
#            wordcloud_real.png  wordcloud_fake.png
# ============================================================

import re
import json
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import nltk
from nltk.corpus import stopwords

# ── captum — optional; graceful fallback if not installed ───
try:
    from captum.attr import LayerIntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Fake News BERT Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }

    .main-title {
        text-align: center; font-size: 2.8rem;
        font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center; color: #4a4e69;
        font-size: 1.15rem; margin-bottom: 1.8rem;
    }
    .card-style {
        background: white; border-radius: 12px;
        padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    .result-box {
        border-radius: 12px; padding: 18px;
        margin: 12px 0; font-weight: 600; font-size: 1.35rem;
    }
    .real-box  { background:#d4edda; border:2px solid #28a745; color:#155724; }
    .fake-box  { background:#f8d7da; border:2px solid #dc3545; color:#721c24; }
    .highlight-container {
        background:#e9ecef; border-radius:8px;
        padding:15px; line-height:2.4;
        font-size:1.02rem; color:black;
        word-break: break-word;
    }
    .credibility-tip {
        background:#fff3cd; border-left:4px solid #ffc107;
        padding:10px 15px; border-radius:5px;
        margin:8px 0; color:#856404;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Cached loaders
# ============================================================
@st.cache_resource
def load_nltk_data():
    nltk.download('stopwords', quiet=True)

@st.cache_resource
def load_model_and_tokenizer():
    model_path = "fake_news_bert"
    tok   = DistilBertTokenizerFast.from_pretrained(model_path)
    mdl   = DistilBertForSequenceClassification.from_pretrained(model_path)
    mdl.eval()
    return tok, mdl

@st.cache_data
def load_metrics():
    with open('fake_news_metrics.json') as f:
        return json.load(f)

# ── load everything ──────────────────────────────────────────
load_nltk_data()
stop_words = set(stopwords.words('english'))

try:
    tokenizer, model = load_model_and_tokenizer()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"❌ Could not load model from `fake_news_bert/`: {e}")

try:
    metrics = load_metrics()
    METRICS_LOADED = True
except Exception:
    METRICS_LOADED = False
    metrics = {}

# ============================================================
# Core functions
# ============================================================
def predict_text(text: str):
    """Return (pred_label, probs_array). 0=Fake, 1=Real."""
    inputs = tokenizer(
        text, return_tensors='pt',
        truncation=True, padding=True, max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred = int(np.argmax(probs))
    return pred, probs


# ── Integrated Gradients explanation ────────────────────────
def _forward_func(input_ids, attention_mask):
    """Wrapper so Captum receives only the tensor it differentiates."""
    emb    = model.distilbert.embeddings.word_embeddings(input_ids)
    output = model(inputs_embeds=emb, attention_mask=attention_mask)
    return output.logits


def explain_text(text: str, pred_class: int):
    """
    Returns list of (token_str, attribution_score) pairs.
    Uses LayerIntegratedGradients on the word-embedding layer.
    """
    if not CAPTUM_AVAILABLE:
        return []

    inputs       = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    input_ids    = inputs['input_ids']          # (1, seq_len)
    attention_mask = inputs['attention_mask']   # (1, seq_len)

    embed_layer  = model.distilbert.embeddings.word_embeddings
    baseline_ids = torch.zeros_like(input_ids)  # [PAD] token baseline

    # ── forward that takes embeddings (not ids) ──────────────
    def fwd_emb(emb_input):
        out = model(inputs_embeds=emb_input, attention_mask=attention_mask)
        return out.logits

    lig = LayerIntegratedGradients(fwd_emb, embed_layer)

    attributions, _ = lig.attribute(
        inputs            = embed_layer(input_ids),    # actual embeddings
        baselines         = embed_layer(baseline_ids), # baseline embeddings
        target            = pred_class,
        return_convergence_delta=True,
        n_steps           = 50,
    )

    # Sum across embedding dim → scalar per token
    attr_scores = attributions.sum(dim=-1).squeeze(0).detach().numpy()
    tokens      = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Filter special tokens
    result = [
        (tok, score)
        for tok, score in zip(tokens, attr_scores)
        if tok not in ('[CLS]', '[SEP]', '[PAD]')
    ]
    return result


def colorize_tokens(token_attr_list):
    """Build HTML with green/red token highlights."""
    if not token_attr_list:
        return ""
    max_abs = max(abs(s) for _, s in token_attr_list) or 1.0
    html = ""
    for token, score in token_attr_list:
        intensity = min(abs(score) / max_abs, 1.0)
        if score >= 0:
            bg = f"rgba(40,167,69,{intensity * 0.75})"   # green → Real
        else:
            bg = f"rgba(220,53,69,{intensity * 0.75})"   # red   → Fake
        # Re-join WordPiece sub-tokens
        display = token[2:] if token.startswith("##") else (" " + token)
        html += (
            f'<span style="background:{bg};color:black;'
            f'padding:2px 5px;border-radius:4px;margin:1px;">'
            f'{display}</span>'
        )
    return html


def credibility_tips(text: str):
    """Simple heuristic checks for sensationalist language."""
    tips = []
    upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if upper_ratio > 0.25:
        tips.append("High ratio of CAPITAL LETTERS — common in sensationalist headlines.")
    if text.count('!') > 3:
        tips.append("Excessive exclamation marks — often used for emotional manipulation.")
    suspicious_phrases = [
        "you won't believe", "shocking truth", "they don't want you to know",
        "conspiracy", "wake up", "mainstream media won't tell you",
        "share before deleted",
    ]
    for phrase in suspicious_phrases:
        if phrase in text.lower():
            tips.append(f'Suspicious phrasing detected: "{phrase}"')
    # Check for all-caps words (beyond normal acronyms)
    caps_words = re.findall(r'\b[A-Z]{4,}\b', text)
    if len(caps_words) > 3:
        tips.append(f"Multiple all-caps words: {', '.join(set(caps_words[:5]))} — often used for emphasis/outrage.")
    return tips

# ============================================================
# Session state
# ============================================================
if 'history' not in st.session_state:
    st.session_state.history = []

# ============================================================
# Header
# ============================================================
st.markdown('<div class="main-title">📰 BERT Fake News Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Explainable AI — see exactly which words influence the verdict</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Main input + result panel
# ============================================================
if MODEL_LOADED:
    with st.container():
        st.markdown('<div class="card-style">', unsafe_allow_html=True)
        col_input, col_result = st.columns([3, 2], gap="large")

        # ── Left: input ──────────────────────────────────────
        with col_input:
            st.subheader("✍️ Paste a news article")

            example_options = [
                "— select an example —",
                "The FDA has approved a new vaccine that could help prevent malaria in children under five.",
                "BREAKING: Shocking conspiracy! You won't believe what they are hiding from you RIGHT NOW!!!",
                "Researchers discover a new species of bioluminescent dolphin in the Pacific Ocean.",
                "WAKE UP! The mainstream media won't tell you the truth about what is really going on.",
            ]
            example = st.selectbox("Or try a built-in example:", example_options)

            default_text = "" if example == example_options[0] else example
            user_text = st.text_area(
                "Article title + body:",
                value=default_text,
                height=240,
                placeholder="Paste any news article text here…",
            )

            analyze_btn = st.button("🔍 Analyse with BERT", use_container_width=True, type="primary")

        # ── Right: result ─────────────────────────────────────
        with col_result:
            if analyze_btn:
                if not user_text.strip():
                    st.warning("Please paste some text first.")
                else:
                    pred, probs = predict_text(user_text)
                    confidence  = float(probs[pred])
                    label_text  = "Real ✅" if pred == 1 else "Fake ❌"

                    # Save to history
                    st.session_state.history.append({
                        'Snippet':     user_text[:180] + ('…' if len(user_text) > 180 else ''),
                        'Verdict':     label_text,
                        'Confidence':  f'{confidence:.2%}',
                    })

                    # Result banner
                    box_cls = "real-box" if pred == 1 else "fake-box"
                    st.markdown(
                        f'<div class="result-box {box_cls}">{label_text}'
                        f'&nbsp; — &nbsp;confidence: {confidence:.1%}</div>',
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"🔹 **Fake:** {probs[0]:.1%} &nbsp;|&nbsp; "
                        f"**Real:** {probs[1]:.1%}"
                    )
                    if confidence < 0.65:
                        st.warning("⚠️ Low confidence — the model is uncertain; treat this result with caution.")

                    # Confidence gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        number={'suffix': "%", 'font': {'size': 28, 'color': 'darkblue'}},
                        title={'text': "Confidence", 'font': {'size': 14}},
                        gauge={
                            'axis':  {'range': [0, 100]},
                            'bar':   {'color': "#28a745" if pred == 1 else "#dc3545"},
                            'steps': [
                                {'range': [0,  50], 'color': '#f0f0f0'},
                                {'range': [50, 100], 'color': '#e0e0e0'},
                            ],
                        },
                    ))
                    fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Word-level explanation (full width below cards) ───────
    if analyze_btn and user_text.strip():
        if CAPTUM_AVAILABLE:
            st.markdown("### 🔍 Word-level Explanation")
            with st.spinner("Computing token attributions…"):
                try:
                    token_attr = explain_text(user_text, pred)
                    if token_attr:
                        html = colorize_tokens(token_attr)
                        st.markdown(
                            f'<div class="highlight-container">{html}</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption("🟢 Green = supports **Real** | 🔴 Red = supports **Fake** | Darker = stronger signal")
                    else:
                        st.info("No token attribution data returned.")
                except Exception as e:
                    st.info(f"Could not compute word importance: {e}")
        else:
            st.info("ℹ️ Install `captum` (`pip install captum`) to enable word-level explanations.")

        # Credibility tips
        tips = credibility_tips(user_text)
        if tips:
            st.markdown("### ⚠️ Credibility Alerts")
            for tip in tips:
                st.markdown(
                    f'<div class="credibility-tip">🚩 {tip}</div>',
                    unsafe_allow_html=True,
                )

# ============================================================
# Tabs: Performance | Word Clouds | History | About
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Model Performance", "🌐 Word Clouds", "🕒 History", "ℹ️ About"]
)

# ── Tab 1: Performance ────────────────────────────────────────
with tab1:
    st.subheader("BERT Model Performance on Test Set")
    if METRICS_LOADED:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Accuracy", f"{metrics['accuracy']:.4f}")
            st.write("**Per-class Classification Report**")
            try:
                report_df = pd.DataFrame(metrics['classification_report']).T
                # Round numeric columns
                num_cols = report_df.select_dtypes(include='number').columns
                report_df[num_cols] = report_df[num_cols].round(4)
                st.dataframe(report_df, use_container_width=True)
            except Exception:
                st.json(metrics['classification_report'])

        with col2:
            st.write("**Confusion Matrix**")
            cm = np.array(metrics['confusion_matrix'])
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Fake', 'Predicted Real'],
                y=['True Fake',      'True Real'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                showscale=True,
            ))
            fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning("Metrics file `fake_news_metrics.json` not found. Run the training script first.")

# ── Tab 2: Word clouds ────────────────────────────────────────
with tab2:
    st.subheader("Word Clouds — Training Data")
    col3, col4 = st.columns(2)
    with col3:
        try:
            st.image('wordcloud_real.png', caption='Real News', use_container_width=True)
        except Exception:
            st.warning("`wordcloud_real.png` not found.")
    with col4:
        try:
            st.image('wordcloud_fake.png', caption='Fake News', use_container_width=True)
        except Exception:
            st.warning("`wordcloud_fake.png` not found.")

# ── Tab 3: History ────────────────────────────────────────────
with tab3:
    st.subheader("Recent Predictions (this session)")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(hist_df, use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet in this session.")

# ── Tab 4: About ──────────────────────────────────────────────
with tab4:
    st.subheader("How it works")
    st.markdown("""
    **Model: DistilBERT (distilbert-base-uncased)**

    DistilBERT is a compact, efficient variant of BERT — 40 % smaller and 60 % faster
    while retaining ~97 % of BERT's language understanding. Here it is fine-tuned on
    a balanced real/fake news dataset to produce a binary classifier.

    **Pipeline**
    1. Article text is cleaned (URLs, HTML, special characters removed).
    2. The tokenizer converts text to subword tokens (max 512 tokens).
    3. DistilBERT produces contextual embeddings, and a classification head predicts
       **Real (1)** or **Fake (0)** with an associated confidence score.

    **Explainability — Integrated Gradients (Captum)**
    Integrated Gradients computes the contribution of each input token to the final
    prediction by interpolating between a neutral baseline (all-padding) and the actual
    input, then integrating the gradient of the output with respect to the embeddings.
    Tokens highlighted in 🟢 green push the model toward *Real*; 🔴 red toward *Fake*.

    **Typical test accuracy:** ~99 % on a balanced dataset of ~44,000 articles.

    **Limitations**
    - The model was trained on a specific dataset and may not generalise to all news domains.
    - Always verify suspicious articles with reputable fact-checking sources.
    """)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4a4e69;'>"
    "Built with ❤️ using Hugging Face Transformers · Captum · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)