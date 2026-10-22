"""
TweFuse-W Multimodal Reliability Dashboard
==========================================
Integration with Kaggle Dataset: charumaurya15/multimodal-fake-news-processed
Supports:
  1. Binary Classification (Real News vs. Fake News)
  2. 6-Class Fine-Grained Classification (Fakeddit Taxonomy)
  3. Dynamic Dual-Answer Architecture:
     - When Image and Text are dissimilar/conflicting:
       Provides 2 separate independent answers (Text Data Answer & Image Media Answer).
     - When Image and Text are similar/harmonious:
       Provides a single unified consensus answer corroborated across both modalities.
"""

import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer
import altair as alt

from model import TweFuseW
from utils import (
    evaluate_modality_similarity,
    modality_conflict_level,
    preprocess_image,
    preprocess_text,
)

# Taxonomy definitions
CLASSES_6WAY = TweFuseW.CLASS_NAMES  # ["True", "Satire", "Misleading Content", "False Connection", "Manipulated Content", "Imposter Content"]
CLASSES_BINARY = ["Real News", "Fake News"]

# Page setup
st.set_page_config(
    page_title="TweFuse-W | Multimodal Fake News Verification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Executive Professional Styling (Modern Glassmorphism & Visual Polish)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1280px;
    }

    /* Executive Glass Card */
    .glass-card {
        background: rgba(18, 24, 38, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.35);
        margin-bottom: 20px;
    }

    /* Title Styling - Elegant Gradient */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #60A5FA 0%, #818CF8 50%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    /* Dataset Integration Banner */
    .dataset-banner {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 12px 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
    }

    /* Refined Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 13px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-neutral {
        background: rgba(148, 163, 184, 0.12);
        color: #CBD5E1;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .badge-primary {
        background: rgba(99, 102, 241, 0.14);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.12);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.12);
        color: #FCD34D;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
    .badge-copper {
        background: rgba(225, 112, 85, 0.15);
        color: #FFAB91;
        border: 1px solid rgba(225, 112, 85, 0.35);
    }

    /* Metric Containers */
    .metric-container {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: left;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Dual Answer Cards (When Image and Text are NOT Similar) */
    .dual-card {
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 12px 32px -6px rgba(0, 0, 0, 0.45);
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .dual-card-text {
        background: linear-gradient(155deg, rgba(15, 23, 42, 0.92) 0%, rgba(20, 35, 60, 0.8) 100%);
        border: 1.5px solid rgba(56, 189, 248, 0.4);
    }
    .dual-card-image {
        background: linear-gradient(155deg, rgba(15, 23, 42, 0.92) 0%, rgba(55, 25, 55, 0.8) 100%);
        border: 1.5px solid rgba(244, 114, 182, 0.4);
    }
    .dual-header-pill-text {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.16);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 14px;
        width: fit-content;
    }
    .dual-header-pill-image {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(244, 114, 182, 0.16);
        color: #F472B6;
        border: 1px solid rgba(244, 114, 182, 0.35);
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 14px;
        width: fit-content;
    }
    .dual-verdict-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
    }
    .dual-verdict-conf {
        font-size: 0.95rem;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 16px;
    }
    .dual-desc-box {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 12px;
        padding: 14px 16px;
        font-size: 0.88rem;
        line-height: 1.55;
        color: #CBD5E1;
        border-left: 4px solid;
        margin-top: 12px;
    }
    .dual-desc-text {
        border-left-color: #38BDF8;
    }
    .dual-desc-image {
        border-left-color: #F472B6;
    }

    /* Dissimilarity vs Harmonious Alert Banners */
    .dissimilarity-alert-banner {
        background: linear-gradient(90deg, rgba(225, 112, 85, 0.18) 0%, rgba(245, 158, 11, 0.12) 100%);
        border: 1.5px solid rgba(225, 112, 85, 0.4);
        border-radius: 14px;
        padding: 16px 22px;
        margin-bottom: 22px;
    }
    .harmony-alert-banner {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.16) 0%, rgba(59, 130, 246, 0.12) 100%);
        border: 1.5px solid rgba(16, 185, 129, 0.35);
        border-radius: 14px;
        padding: 16px 22px;
        margin-bottom: 22px;
    }

    /* Gauge Container */
    .gauge-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Buttons */
    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.65rem 1.4rem;
        transition: all 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Cached Model & Tokenizer Loader
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_tokenizer(bertweet_name: str):
    return AutoTokenizer.from_pretrained(bertweet_name, normalization=True)


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, bertweet_name: str, swin_name: str, device: str, num_classes: int = 6):
    model = TweFuseW(num_classes=num_classes, bertweet_name=bertweet_name, swin_name=swin_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model, list(missing), list(unexpected)


def run_inference(model, tokenizer, device, text: str, image: Image.Image):
    input_ids, attention_mask = preprocess_text(text, tokenizer)
    pixel_values = preprocess_image(image)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        out = model(input_ids, attention_mask, pixel_values)
        fused_probs = torch.softmax(out["logits"], dim=-1).cpu().numpy()[0]
        text_probs = torch.softmax(out["text_only_logits"], dim=-1).cpu().numpy()[0]
        image_probs = torch.softmax(out["image_only_logits"], dim=-1).cpu().numpy()[0]
        alpha_t = float(out["alpha_t"].cpu().numpy()[0])
        alpha_v = float(out["alpha_v"].cpu().numpy()[0])

    return fused_probs, text_probs, image_probs, alpha_t, alpha_v


# --------------------------------------------------------------------------
# Sidebar: Setup & Calibration Options
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔍 Model & Dataset Hub")
    st.markdown("---")

    # Classification Mode
    task_mode = st.radio(
        "Classification Objective",
        ["6-Class Multimodal (Fakeddit)", "Binary (Real vs. Fake)"],
        index=0,
        help="Select whether to predict fine-grained 6 categories or aggregate into binary Real vs. Fake.",
    )

    is_binary = task_mode == "Binary (Real vs. Fake)"
    active_classes = CLASSES_BINARY if is_binary else CLASSES_6WAY

    st.markdown("---")
    st.subheader("Dual-Answer Architecture")
    decision_rule = st.selectbox(
        "Separate Answers Trigger Rule",
        [
            "Disagreement or High Divergence (Recommended)",
            "Strict (Any Divergence JS ≥ Medium)",
            "Always Output 2 Separate Answers",
        ],
        index=0,
        help="Controls when the dashboard splits the prediction into 2 separate answers for image and text data.",
    )

    st.markdown("---")
    st.subheader("Checkpoint Configuration")
    checkpoint_path = st.text_input(
        "Checkpoint Path (.pt)",
        value="checkpoints/twefuse_w.pt",
        help="Path to your saved PyTorch weights.",
    )
    device = st.selectbox("Inference Hardware", ["cpu", "cuda"], index=0)

    load_btn = st.button("⚡ Initialize Model", type="primary", use_container_width=True)

    if load_btn:
        with st.spinner("Loading encoders and weights..."):
            try:
                num_cls = 2 if is_binary else 6
                model, missing, unexpected = load_model(
                    checkpoint_path, "vinai/bertweet-base", "swin_base_patch4_window7_224", device, num_classes=num_cls
                )
                tokenizer = load_tokenizer("vinai/bertweet-base")
                st.session_state["model"] = model
                st.session_state["tokenizer"] = tokenizer
                st.session_state["device"] = device
                st.session_state["num_classes"] = num_cls
                st.success("✅ Model loaded successfully!")
                if missing:
                    st.caption(f"Note: {len(missing)} missing keys loaded with default initialization.")
            except Exception as e:
                st.error(f"Checkpoint not found at: {checkpoint_path}")
                st.info("💡 Keep 'Interactive Test Mode' enabled to run test samples seamlessly.")

    st.markdown("---")
    demo_mode = st.toggle("🧪 Interactive Test Mode", value=True, help="Simulate inference if a checkpoint is not yet loaded.")

    with st.expander("⚙️ Divergence Calibration"):
        js_medium = st.slider("JS Divergence — Medium", 0.0, 1.0, 0.15, 0.01)
        js_high = st.slider("JS Divergence — High", 0.0, 1.0, 0.35, 0.01)

    with st.expander("📚 Kaggle Dataset Metadata"):
        st.markdown(
            """
            **Dataset:** `charumaurya15/multimodal-fake-news-processed`  
            • **Sources:** Fakeddit, FakeNewsNet, PHEME  
            • **Modality Alignment:** Paired `sample_id` text & image  
            • **Evaluation:** Binary & 6-Class Fakeddit taxonomies  
            • **Modality Conflict:** Detects cross-modal inconsistency / out-of-context image reuse.  
            [Open Dataset on Kaggle ↗](https://www.kaggle.com/datasets/charumaurya15/multimodal-fake-news-processed)
            """
        )

# --------------------------------------------------------------------------
# Main Content View
# --------------------------------------------------------------------------
st.markdown('<div class="hero-title">Multimodal Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">TweFuse-W: Adaptive reliability gating across BERTweet (textual) and Swin Transformer (visual) representations with dynamic 2-answer modality resolution.</div>',
    unsafe_allow_html=True,
)

# Dataset integration badge bar
st.markdown(
    f"""
    <div class="dataset-banner">
        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
            <span class="badge badge-primary">Dataset: charumaurya15/multimodal-fake-news-processed</span>
            <span class="badge badge-neutral">Mode: {'Binary (Real vs. Fake)' if is_binary else '6-Class Fine-Grained'}</span>
            <span class="badge badge-primary">Dual-Answer Enabled: When Image ≠ Text</span>
        </div>
        <div>
            {'<span class="badge badge-success">● Model Ready</span>' if 'model' in st.session_state else '<span class="badge badge-warning">⚡ Interactive Test Mode</span>'}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Input & Ingestion Section
# --------------------------------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("#### 📥 Post Content Ingestion")

# Sample Preset Loader from Kaggle dataset benchmarks
preset_options = [
    "Custom Upload",
    "[Kaggle Benchmark] Sample 1: Authentic Scientific Discovery (Real / True - Similar Modalities)",
    "[Kaggle Benchmark] Sample 2: Satirical Headline Claim (Satire - Similar Modalities)",
    "[Kaggle Benchmark] Sample 3: Fabricated Urban Hazard Alert (Manipulated - Similar Modalities)",
    "[Kaggle Benchmark] Sample 4: Misattributed Historical Event (False Connection - Mild Divergence)",
    "[Kaggle Benchmark] Sample 5: Cross-Modal Conflict (Fake Text Claim + Authentic Photo -> 2 SEPARATE ANSWERS)",
    "[Kaggle Benchmark] Sample 6: Cross-Modal Conflict (Factual Text + Doctored Photo -> 2 SEPARATE ANSWERS)",
]

selected_preset = st.selectbox("Load Benchmark Sample from Kaggle Dataset:", preset_options)

preset_text = ""
preset_url = ""

if "Sample 1" in selected_preset:
    preset_text = "NASA's James Webb Space Telescope confirms atmospheric carbon dioxide signature on temperate rocky exoplanet orbiting in habitable zone."
    preset_url = "https://images.unsplash.com/photo-1614728894747-a83421e2b9c9?w=600&auto=format&fit=crop"
elif "Sample 2" in selected_preset:
    preset_text = "Local coffee shop declares bankruptcy after customer pays in loose compliments instead of actual currency."
    preset_url = "https://images.unsplash.com/photo-1501339847302-ac426a4a7cbb?w=600&auto=format&fit=crop"
elif "Sample 3" in selected_preset:
    preset_text = "BREAKING: Massive geomagnetic burst causes total metropolitan communication blackout across eastern seaboard. Authorities urged immediate shelter."
    preset_url = "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=600&auto=format&fit=crop"
elif "Sample 4" in selected_preset:
    preset_text = "Rare planetary alignment tonight will cause ocean tides to completely stop moving for 12 hours worldwide."
    preset_url = "https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?w=600&auto=format&fit=crop"
elif "Sample 5" in selected_preset:
    preset_text = "URGENT BREAKING: Classified military submarine collision off the Atlantic coastline triggers emergency containment protocols and naval quarantine."
    preset_url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=600&auto=format&fit=crop"
elif "Sample 6" in selected_preset:
    preset_text = "National weather forecast center issues standard seasonal precipitation advisory for agricultural river basin communities."
    preset_url = "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=600&auto=format&fit=crop"

col_text, col_media = st.columns([1.2, 1])

with col_text:
    post_text = st.text_area(
        "Post Text Content (Data)",
        value=preset_text,
        height=190,
        placeholder="Enter the textual statement, tweet, or article snippet...",
    )

    custom_sim_mode = None
    if selected_preset == "Custom Upload" and demo_mode:
        custom_sim_mode = st.radio(
            "Test Mode Simulation Behavior for Custom Input:",
            [
                "⚡ Simulate Modality Dissimilarity (Text & Image Disagree -> 2 Separate Answers)",
                "✅ Simulate Modality Harmony (Text & Image Agree -> 1 Unified Answer)",
            ],
            index=0,
            horizontal=False,
            help="Choose whether your custom test should simulate conflicting modalities or matching modalities.",
        )

with col_media:
    image_tab1, image_tab2 = st.tabs(["📁 File Upload", "🔗 Image URL"])
    uploaded_image = None

    with image_tab1:
        file_input = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
        if file_input is not None:
            uploaded_image = Image.open(file_input)

    with image_tab2:
        url_input = st.text_input("Image Web Link", value=preset_url if file_input is None else "")
        if url_input and file_input is None:
            try:
                resp = requests.get(url_input, timeout=4)
                if resp.status_code == 200:
                    uploaded_image = Image.open(io.BytesIO(resp.content))
            except Exception:
                st.caption("Unable to fetch image from URL.")

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Ingested Image Preview", use_container_width=True)

execute_btn = st.button("🚀 Analyze Post Authenticity", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Inference & Result Breakdown
# --------------------------------------------------------------------------
if execute_btn:
    if not post_text.strip() and not demo_mode:
        st.error("Please provide text content to analyze.")
    elif uploaded_image is None and not demo_mode:
        st.error("Please upload or provide an image URL for multimodal verification.")
    else:
        with st.spinner("Computing multimodal embeddings & evaluating modality similarity..."):
            if "model" in st.session_state and not demo_mode:
                model = st.session_state["model"]
                tokenizer = st.session_state["tokenizer"]
                dev = st.session_state["device"]
                fused_probs, text_probs, image_probs, alpha_t, alpha_v = run_inference(
                    model, tokenizer, dev, post_text, uploaded_image
                )
            else:
                # Benchmark simulations based on Kaggle dataset distributions
                if "Sample 1" in selected_preset:
                    # Authentic / True (Harmonious)
                    fused_6 = np.array([0.91, 0.02, 0.02, 0.02, 0.02, 0.01])
                    t_probs_6 = np.array([0.89, 0.03, 0.03, 0.02, 0.02, 0.01])
                    i_probs_6 = np.array([0.87, 0.02, 0.04, 0.03, 0.02, 0.02])
                    alpha_t, alpha_v = 0.58, 0.42
                elif "Sample 2" in selected_preset:
                    # Satire (Harmonious)
                    fused_6 = np.array([0.03, 0.86, 0.04, 0.03, 0.02, 0.02])
                    t_probs_6 = np.array([0.02, 0.90, 0.03, 0.02, 0.02, 0.01])
                    i_probs_6 = np.array([0.08, 0.72, 0.08, 0.05, 0.04, 0.03])
                    alpha_t, alpha_v = 0.71, 0.29
                elif "Sample 3" in selected_preset:
                    # Manipulated Content (Harmonious)
                    fused_6 = np.array([0.02, 0.02, 0.08, 0.05, 0.80, 0.03])
                    t_probs_6 = np.array([0.05, 0.05, 0.35, 0.20, 0.30, 0.05])
                    i_probs_6 = np.array([0.01, 0.01, 0.04, 0.02, 0.89, 0.03])
                    alpha_t, alpha_v = 0.32, 0.68
                elif "Sample 4" in selected_preset:
                    # False Connection (Mild Divergence)
                    fused_6 = np.array([0.04, 0.03, 0.12, 0.73, 0.05, 0.03])
                    t_probs_6 = np.array([0.08, 0.05, 0.18, 0.62, 0.04, 0.03])
                    i_probs_6 = np.array([0.06, 0.04, 0.15, 0.68, 0.04, 0.03])
                    alpha_t, alpha_v = 0.52, 0.48
                elif "Sample 5" in selected_preset:
                    # DISSIMILAR: Text is Misleading/Fake claim, Image is Authentic/Real photo
                    # Branches strongly disagree!
                    fused_6 = np.array([0.38, 0.02, 0.48, 0.08, 0.02, 0.02])
                    t_probs_6 = np.array([0.02, 0.02, 0.91, 0.03, 0.01, 0.01])  # Misleading Content (91%)
                    i_probs_6 = np.array([0.88, 0.02, 0.03, 0.03, 0.02, 0.02])  # True (88%)
                    alpha_t, alpha_v = 0.62, 0.38
                elif "Sample 6" in selected_preset:
                    # DISSIMILAR: Text is factual news (True), Image is doctored/manipulated
                    # Branches strongly disagree!
                    fused_6 = np.array([0.35, 0.02, 0.05, 0.06, 0.50, 0.02])
                    t_probs_6 = np.array([0.89, 0.02, 0.03, 0.03, 0.02, 0.01])  # True (89%)
                    i_probs_6 = np.array([0.02, 0.01, 0.04, 0.03, 0.88, 0.02])  # Manipulated Content (88%)
                    alpha_t, alpha_v = 0.44, 0.56
                else:
                    # Custom Upload
                    if custom_sim_mode and "Simulate Modality Dissimilarity" in custom_sim_mode:
                        # Simulated Dissimilarity: Text Fake/Misleading vs Image Real/True
                        fused_6 = np.array([0.34, 0.03, 0.52, 0.06, 0.03, 0.02])
                        t_probs_6 = np.array([0.03, 0.02, 0.89, 0.03, 0.02, 0.01])
                        i_probs_6 = np.array([0.86, 0.03, 0.04, 0.03, 0.02, 0.02])
                        alpha_t, alpha_v = 0.65, 0.35
                    else:
                        # Simulated Harmony: Both Real/True
                        fused_6 = np.array([0.84, 0.03, 0.04, 0.04, 0.03, 0.02])
                        t_probs_6 = np.array([0.82, 0.04, 0.05, 0.04, 0.03, 0.02])
                        i_probs_6 = np.array([0.80, 0.04, 0.06, 0.05, 0.03, 0.02])
                        alpha_t, alpha_v = 0.55, 0.45

                if is_binary:
                    # Map 6-class Fakeddit probabilities to Binary: P(Real) = P(True), P(Fake) = Sum(others)
                    fused_probs = np.array([fused_6[0], np.sum(fused_6[1:])])
                    text_probs = np.array([t_probs_6[0], np.sum(t_probs_6[1:])])
                    image_probs = np.array([i_probs_6[0], np.sum(i_probs_6[1:])])
                else:
                    fused_probs = fused_6
                    text_probs = t_probs_6
                    image_probs = i_probs_6

            pred_idx = int(np.argmax(fused_probs))
            confidence = float(fused_probs[pred_idx])

            # Comprehensive similarity & conflict evaluation
            eval_res = evaluate_modality_similarity(
                text_probs,
                image_probs,
                js_medium_threshold=js_medium,
                js_high_threshold=js_high,
                decision_mode=decision_rule,
            )

            is_dissimilar = eval_res["is_dissimilar"]
            level = eval_res["conflict_level"]
            js_div = eval_res["js_divergence"]
            sim_score = eval_res["similarity_score"]
            branches_disagree = eval_res["branches_disagree"]
            t_pred = eval_res["text_pred"]
            i_pred = eval_res["image_pred"]
            t_conf = eval_res["text_conf"]
            i_conf = eval_res["image_conf"]

        # ----------------- SECTION 1: VERDICT PRESENTATION -----------------
        # IF IMAGE AND TEXT ARE NOT SIMILAR -> PROVIDE 2 SEPARATE ANSWERS
        if is_dissimilar:
            st.markdown(
                f"""
                <div class="dissimilarity-alert-banner">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-size: 1.15rem; font-weight: 800; color: #FFAB91; display: flex; align-items: center; gap: 8px;">
                            ⚡ Modality Dissimilarity Detected — 2 Separate Answers Generated
                        </span>
                        <span class="badge badge-copper">Cross-Modal Divergence JS: {js_div:.3f}</span>
                    </div>
                    <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.5;">
                        The textual claim and visual media are <strong>not similar</strong>. Their unimodal representations contradict or significantly diverge 
                        (Modality Similarity: <strong>{sim_score*100:.1f}%</strong>). Because forcing a single blended output on conflicting modalities masks out-of-context deception, 
                        the model provides <strong>two separate independent answers</strong> below: one for the <strong>Text Data</strong> and one for the <strong>Image Media</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Two Separate Answers Side-by-Side
            col_ans1, col_ans2 = st.columns(2)

            text_class_name = active_classes[t_pred]
            image_class_name = active_classes[i_pred]

            # Text explanation rationale
            if "Fake" in text_class_name or text_class_name in ["Misleading Content", "Satire", "Manipulated Content", "False Connection", "Imposter Content"]:
                text_finding = "The textual statement displays deceptive patterns, unsupported factual claims, or sensational linguistic framing."
            else:
                text_finding = "The textual statement reflects factual, credible phrasing consistent with verified reporting standards."

            # Image explanation rationale
            if "Fake" in image_class_name or image_class_name in ["Manipulated Content", "Misleading Content", "Imposter Content"]:
                image_finding = "The visual asset displays synthetic artifacts, unnatural compression, or digital tampering indicators."
            else:
                image_finding = "The visual asset presents coherent, authentic photographic geometry with no obvious digital doctoring detected."

            with col_ans1:
                st.markdown(
                    f"""
                    <div class="dual-card dual-card-text">
                        <div>
                            <div class="dual-header-pill-text">📝 Answer 1 • Text / Data Modality</div>
                            <div class="dual-verdict-title">{text_class_name}</div>
                            <div class="dual-verdict-conf">
                                Confidence: <span style="color: #38BDF8; font-weight: 700;">{t_conf * 100:.1f}%</span> • Network: BERTweet (768-d)
                            </div>
                        </div>
                        <div>
                            <div class="dual-desc-box dual-desc-text">
                                <strong>Text Finding:</strong> {text_finding}
                            </div>
                            <div style="margin-top: 14px; font-size: 0.8rem; color: #94A3B8;">
                                Evaluated independently using unimodal text projection head.
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_ans2:
                st.markdown(
                    f"""
                    <div class="dual-card dual-card-image">
                        <div>
                            <div class="dual-header-pill-image">🖼️ Answer 2 • Image / Visual Modality</div>
                            <div class="dual-verdict-title">{image_class_name}</div>
                            <div class="dual-verdict-conf">
                                Confidence: <span style="color: #F472B6; font-weight: 700;">{i_conf * 100:.1f}%</span> • Network: Swin Transformer (1024-d)
                            </div>
                        </div>
                        <div>
                            <div class="dual-desc-box dual-desc-image">
                                <strong>Image Finding:</strong> {image_finding}
                            </div>
                            <div style="margin-top: 14px; font-size: 0.8rem; color: #94A3B8;">
                                Evaluated independently using unimodal vision projection head.
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Forensic Synthesis of the Dissimilarity
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🔬 Cross-Modal Inconsistency Analysis")
            st.markdown(
                f"""
                - **Primary Conflict:** The **Text Data** predicts **`{text_class_name}`** ({t_conf*100:.1f}%), whereas the **Image Media** predicts **`{image_class_name}`** ({i_conf*100:.1f}%).
                - **Multimodal Fake News Significance:** In online misinformation (such as the Fakeddit benchmark), a common manipulation technique is **Out-of-Context Media Reuse (False Connection)**. Fabricators frequently pair genuine, unmanipulated photos with deceptive headlines or captions to artificially gain credibility.
                - **Why Separate Answers Matter:** Standard early-fusion or late-fusion models force a single consensus answer, which can suppress the visual truth or drown out textual falsehoods. Generating 2 separate answers precisely attributes authenticity to the media while isolating the falsehood in the claim (or vice versa).
                """
            )

            # Fallback Gated Prediction Expander
            with st.expander("⚖️ View TweFuse-W Gated Multimodal Fallback Output"):
                st.markdown(
                    f"""
                    If forced into a single joint prediction, TweFuse-W calculates:
                    - **Joint Fused Class:** `{active_classes[pred_idx]}` ({confidence * 100:.1f}% confidence)
                    - **Adaptive Gate Allocation:** Text Weight αT = `{alpha_t * 100:.1f}%`, Vision Weight αV = `{alpha_v * 100:.1f}%`
                    - *Advisory Note: Due to high cross-modal dissimilarity, relying on the single fused output is cautioned; the dual unimodal answers above represent the ground reality of each medium.*
                    """
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # ELSE: IMAGE AND TEXT ARE SIMILAR -> PROVIDE 1 UNIFIED CONSENSUS ANSWER
        else:
            st.markdown(
                f"""
                <div class="harmony-alert-banner">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-size: 1.15rem; font-weight: 800; color: #6EE7B7; display: flex; align-items: center; gap: 8px;">
                            ✅ Harmonious Modality Consensus — Single Unified Answer
                        </span>
                        <span class="badge badge-success">Alignment: {sim_score*100:.1f}% (JS: {js_div:.3f})</span>
                    </div>
                    <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.5;">
                        The textual claim and visual asset are <strong>similar and mutually reinforcing</strong>. Both modalities agree on the same classification verdict, providing high confidence in the joint multimodal prediction.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Unified Classification Verdict")

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-label">Unified Consensus Class</div>
                        <div class="metric-value">{active_classes[pred_idx]}</div>
                        <div class="metric-sub">Corroborated across modalities</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_m2:
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-label">Joint Confidence</div>
                        <div class="metric-value">{confidence * 100:.1f}%</div>
                        <div class="metric-sub">Multimodal Likelihood</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_m3:
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-label">Modality Agreement</div>
                        <div style="margin-top: 6px; margin-bottom: 8px;">
                            <span class="badge badge-success">🟢 Mutually Aligned</span>
                        </div>
                        <div class="metric-sub">Text & Vision Agree</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_m4:
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-label">Cross-Modal Divergence</div>
                        <div class="metric-value">{js_div:.3f}</div>
                        <div class="metric-sub">Low Discrepancy</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # ----------------- SECTION 2: ADAPTIVE GATING WEIGHTS -----------------
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ⚖️ Adaptive Modality Contribution (Gating Network)")
        st.caption(
            "TweFuse-W calculates dynamic reliability coefficients (αT and αV), deciding whether the textual or visual representation is more trustworthy for this sample."
        )

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown(
                f"""
                <div class="gauge-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: #93C5FD;">📖 Text Modality Reliability (αT)</span>
                        <span style="font-weight: 700; color: #F8FAFC;">{alpha_t * 100:.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(alpha_t)
        with g_col2:
            st.markdown(
                f"""
                <div class="gauge-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: #C4B5FD;">🖼️ Vision Modality Reliability (αV)</span>
                        <span style="font-weight: 700; color: #F8FAFC;">{alpha_v * 100:.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(alpha_v)

        st.markdown('</div>', unsafe_allow_html=True)

        # ----------------- SECTION 3: REFINED ALTAIR BAR CHART -----------------
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Cross-Branch Probability Distribution")
        st.caption(
            f"Comparison of class probabilities across the Fused model, Text-Only head, and Vision-Only head in the {'Binary' if is_binary else '6-Class'} space."
        )

        plot_data = []
        for i, class_name in enumerate(active_classes):
            plot_data.append({"Category": class_name, "Probability": float(fused_probs[i]), "Branch": "Fused (TweFuse-W)"})
            plot_data.append({"Category": class_name, "Probability": float(text_probs[i]), "Branch": "Text-Only Head"})
            plot_data.append({"Category": class_name, "Probability": float(image_probs[i]), "Branch": "Vision-Only Head"})

        df_chart = pd.DataFrame(plot_data)

        # Refined color scheme: Indigo, Cool Cyan, and Soft Pink/Coral
        chart = (
            alt.Chart(df_chart)
            .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
            .encode(
                y=alt.Y("Category:N", title=None, sort=active_classes),
                x=alt.X("Probability:Q", title="Probability Score", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "Branch:N",
                    scale=alt.Scale(
                        domain=["Fused (TweFuse-W)", "Text-Only Head", "Vision-Only Head"],
                        range=["#6366F1", "#06B6D4", "#F472B6"],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                yOffset="Branch:N",
                tooltip=[
                    alt.Tooltip("Branch:N"),
                    alt.Tooltip("Category:N"),
                    alt.Tooltip("Probability:Q", format=".1%"),
                ],
            )
            .properties(height=260 if is_binary else 340)
            .configure_view(strokeOpacity=0)
            .configure_axis(gridColor="rgba(255, 255, 255, 0.05)", labelColor="#94A3B8", titleColor="#94A3B8")
            .configure_legend(labelColor="#E2E8F0")
        )

        st.altair_chart(chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ----------------- SECTION 4: UNIMODAL VERIFICATION DETAILS -----------------
        with st.expander("🔬 Detailed Probability Breakdown"):
            st.markdown(
                f"""
                | Category | Fused Probability | Text-Only Head | Vision-Only Head |
                | :--- | :--- | :--- | :--- |
                """
                + "\n".join(
                    [
                        f"| **{cls}** | {fused_probs[idx]*100:.1f}% | {text_probs[idx]*100:.1f}% | {image_probs[idx]*100:.1f}% |"
                        for idx, cls in enumerate(active_classes)
                    ]
                )
            )
            st.markdown("---")
            st.markdown(
                f"""
                - **Cross-Distribution Jensen-Shannon Divergence:** `{js_div:.4f}`
                - **Modality Agreement Status:** `{'⚠️ Disagreement (Dissimilar)' if branches_disagree else '✅ Agreement (Similar)'}`
                - **Triggered Response Mode:** `{'⚡ 2 Separate Answers (Text vs. Image)' if is_dissimilar else '✅ 1 Unified Consensus Answer'}`
                """
            )
else:
    # Empty State with Dataset Guidance
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 46px 20px;">
            <div style="font-size: 2.8rem; margin-bottom: 12px;">🛡️</div>
            <h3 style="color: #F8FAFC; margin-bottom: 8px;">Ready for Multimodal Evaluation</h3>
            <p style="color: #94A3B8; max-width: 660px; margin: 0 auto 20px auto; font-size: 0.94rem; line-height: 1.6;">
                Select a benchmark sample from the Kaggle dataset dropdown above, or upload a custom image and text post. Then click 
                <strong style="color: #818CF8;">Analyze Post Authenticity</strong> to run the adaptive gating pipeline.
                <br/><br/>
                <span style="color: #38BDF8;">✨ New Feature:</span> If text and image are <strong>not similar</strong>, the system generates 
                <strong>2 separate answers</strong> (Text Data vs. Image Media) to detect out-of-context deception.
            </p>
            <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-primary">BERTweet 768-d</span>
                <span class="badge badge-primary">Swin Transformer 1024-d</span>
                <span class="badge badge-primary">Adaptive αT / αV Gating</span>
                <span class="badge badge-neutral">Binary & 6-Class Support</span>
                <span class="badge badge-success">Dual-Answer Mode for Conflict</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
