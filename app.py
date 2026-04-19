"""
app.py
Main Streamlit application for Retinal Cardiovascular Risk Prediction.
Provides image upload, clinical data inputs, risk prediction, Grad-CAM visualization,
and multi-image batch validation support.

Flow:
  1. Upload Page – upload multiple images + set clinical data
  2. Loading Screen – animated processing screen (3-5 seconds)
  3. Results Page – images classified into High Risk / Low Risk with expandable Grad-CAM
"""

import io
import time
import streamlit as st
import numpy as np
from PIL import Image

from model_loader import load_model
from predictor import preprocess_image, prepare_clinical_data, predict, interpret_risk, is_retinal_image
from gradcam import generate_gradcam, overlay_heatmap


# ─── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinaRisk — Cardiovascular Risk Prediction",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #888;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Score card */
    .score-card {
        text-align: center;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #ffffff11;
    }
    .score-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #fff;
    }
    .score-subtitle {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    /* Interpretation text */
    .interpretation {
        text-align: center;
        font-size: 1rem;
        color: #aaa;
        padding: 0.5rem 1rem;
        border-left: 3px solid #667eea;
        margin: 0.5rem auto;
        max-width: 600px;
        background: #667eea11;
        border-radius: 0 8px 8px 0;
    }

    /* Section divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea55, transparent);
        margin: 1.5rem 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #667eea;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #555;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
    }

    /* ─── Loading Screen ────────────────────────────────────── */
    @keyframes pulse-ring {
        0%   { transform: scale(0.5); opacity: 0.8; }
        80%, 100% { transform: scale(2.2); opacity: 0; }
    }
    @keyframes pulse-dot {
        0%, 100% { transform: scale(1); }
        50%      { transform: scale(1.15); }
    }
    @keyframes float-text {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(-6px); }
    }
    @keyframes scan-line {
        0%   { top: 0%; }
        100% { top: 100%; }
    }

    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 5rem 2rem;
    }
    .pulse-wrapper {
        position: relative;
        width: 120px; height: 120px;
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 2rem;
    }
    .pulse-wrapper .ring {
        position: absolute;
        width: 100%; height: 100%;
        border-radius: 50%;
        border: 3px solid #667eea;
        animation: pulse-ring 1.5s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }
    .pulse-wrapper .ring:nth-child(2) { animation-delay: 0.4s; }
    .pulse-wrapper .ring:nth-child(3) { animation-delay: 0.8s; }
    .pulse-wrapper .icon {
        font-size: 3rem;
        animation: pulse-dot 1.5s ease-in-out infinite;
        z-index: 2;
    }
    .loading-title {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: float-text 2s ease-in-out infinite;
        margin-bottom: 0.5rem;
    }
    .loading-sub {
        color: #888;
        font-size: 0.95rem;
    }

    /* ─── Result cards ──────────────────────────────────────── */
    .risk-group-header {
        font-size: 1.4rem;
        font-weight: 700;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .risk-group-header.high {
        color: #FF4B4B;
        background: linear-gradient(135deg, #FF4B4B11, #FF4B4B08);
        border-left: 4px solid #FF4B4B;
    }
    .risk-group-header.low {
        color: #00C853;
        background: linear-gradient(135deg, #00C85311, #00C85308);
        border-left: 4px solid #00C853;
    }

    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        border: 1px solid #ffffff0d;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.12);
    }

    .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Heatmap toggle button style */
    .heatmap-section {
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid #ffffff0d;
    }

    /* Progress steps */
    .step-bar {
        display: flex;
        justify-content: center;
        gap: 2rem;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    .step {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.9rem;
        color: #555;
    }
    .step.active {
        color: #667eea;
        font-weight: 600;
    }
    .step.done {
        color: #00C853;
    }
    .step-num {
        width: 28px; height: 28px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        border: 2px solid currentColor;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session-state defaults ─────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "upload"          # "upload" | "loading" | "results"
if "results" not in st.session_state:
    st.session_state.results = []             # list of result dicts
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = []       # raw bytes for processing


# ─── Load Model (cached, runs once) ─────────────────────────────────────────────
try:
    with st.spinner("🔧 Loading AI model — this may take a moment on first run..."):
        model = load_model()
except (FileNotFoundError, RuntimeError) as e:
    st.error(f"❌ **Model Loading Error**\n\n{e}")
    st.stop()


# ─── Helper: step indicator ─────────────────────────────────────────────────────
def render_step_bar(current):
    """Render a 3-step progress bar."""
    steps = [("1", "Upload"), ("2", "Processing"), ("3", "Results")]
    html = '<div class="step-bar">'
    for idx, (num, label) in enumerate(steps):
        if idx + 1 < current:
            cls = "done"
            icon = "✓"
        elif idx + 1 == current:
            cls = "active"
            icon = num
        else:
            cls = ""
            icon = num
        html += f'<div class="step {cls}"><span class="step-num">{icon}</span>{label}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─── Helper: risk label badge ───────────────────────────────────────────────────
def display_risk_label(label, emoji, ui_color):
    """Dynamic styling for the risk label using the computed color."""
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 1.8rem; font-weight: 700; padding: 0.8rem 1.5rem;
                    border-radius: 12px; margin: 0.5rem 0;
                    background: linear-gradient(135deg, {ui_color}33, {ui_color}11);
                    color: {ui_color}; border: 2px solid {ui_color}44;">
            {emoji} {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULT CARD RENDERER (must be defined before page routing calls it)
# ═══════════════════════════════════════════════════════════════════════════════
def _render_result_card(r, key_prefix):
    """Render a single result card with risk info and expandable Grad-CAM."""

    risk = r["risk"]

    with st.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        # ── Top row: image name + risk badge ─────────────────────────────
        col_info, col_score = st.columns([3, 1])

        with col_info:
            st.markdown(f"##### 📄 {r['name']}")
            display_risk_label(risk["label"], risk["emoji"], risk["color"])

        with col_score:
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-value" style="color: {risk['color']}">{risk['percent']:.1f}%</div>
                    <div class="score-subtitle">Risk Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Confidence bar
        st.progress(float(r["probability"]), text=f"Risk Measure: {risk['percent']:.2f}%")

        # Interpretation
        st.markdown(
            f'<div class="interpretation">💡 {risk["interpretation"]}</div>',
            unsafe_allow_html=True,
        )

        # ── Original image (always visible) ──────────────────────────────
        st.image(r["original_image"], caption="Retinal Scan", use_container_width=False, width=320)

        # ── Grad-CAM heatmap (expandable on demand) ──────────────────────
        if r.get("gradcam_image") is not None:
            with st.expander("🔥 View Grad-CAM Heatmap", expanded=False):
                col_orig, col_heat = st.columns(2)
                with col_orig:
                    st.markdown("**Original**")
                    st.image(r["original_image"], use_container_width=True)
                with col_heat:
                    st.markdown("**Grad-CAM Overlay**")
                    st.image(r["gradcam_image"], use_container_width=True)

                # Download button
                gradcam_pil = Image.fromarray(r["gradcam_image"])
                buf = io.BytesIO()
                gradcam_pil.save(buf, format="PNG")
                buf.seek(0)
                st.download_button(
                    label="⬇️ Download Grad-CAM Overlay",
                    data=buf,
                    file_name=f"gradcam_{r['name']}",
                    mime="image/png",
                    use_container_width=True,
                    key=f"dl_{key_prefix}",
                )
        else:
            st.info("Grad-CAM visualization unavailable for this image.")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 – UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "upload":

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>👁️ RetinaRisk</h1>
        <p>AI-Powered Cardiovascular Risk Prediction from Retinal Fundus Images</p>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    render_step_bar(1)

    # ── Sidebar: Clinical Data ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🩺 Clinical Data")
        st.caption("Adjust the clinical parameters below. These apply to all uploaded images.")
        st.markdown("---")

        bp = st.slider(
            "Systolic Blood Pressure (mmHg)",
            min_value=80, max_value=200, value=120, step=1,
            help="Normal: 90-120 mmHg. Hypertension: >140 mmHg.",
        )
        cholesterol = st.slider(
            "Total Cholesterol (mg/dL)",
            min_value=100, max_value=400, value=200, step=5,
            help="Desirable: <200 mg/dL. High: >240 mg/dL.",
        )
        bmi = st.slider(
            "Body Mass Index (BMI)",
            min_value=15.0, max_value=45.0, value=25.0, step=0.5,
            help="Normal: 18.5-24.9. Overweight: 25-29.9. Obese: ≥30.",
        )
        diabetes = st.selectbox(
            "Diabetes Status",
            options=[0, 1],
            format_func=lambda x: "No Diabetes" if x == 0 else "Diabetic",
            help="Select current diabetes status.",
        )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#666; font-size:0.8rem;'>"
            "These values are combined with retinal analysis for a multi-modal prediction."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Upload Section ───────────────────────────────────────────────────────────
    st.markdown("### 📤 Upload Retinal Fundus Image(s)")

    uploaded_files = st.file_uploader(
        "Choose retinal fundus images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload high-quality retinal fundus photographs. Non-retinal images will be rejected.",
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.markdown(
            """
            <div style="text-align:center; padding: 4rem 2rem; color: #666;">
                <p style="font-size: 3rem;">👁️⚕️</p>
                <p style="font-size: 1.2rem; font-weight: 500;">
                    Upload one or more retinal fundus images to analyze risk
                </p>
                <p style="font-size: 0.9rem; color: #555;">
                    Supported formats: JPG, JPEG, PNG (Batch upload supported)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)}** image(s) selected")

        # Big analysis button
        if st.button("🚀  Analyze All Images", type="primary", use_container_width=True):
            # Store everything needed for processing
            file_data = []
            for uf in uploaded_files:
                uf.seek(0)
                file_data.append({"name": uf.name, "bytes": uf.read()})

            st.session_state.uploaded_data = file_data
            st.session_state.clinical = {
                "bp": bp, "cholesterol": cholesterol, "bmi": bmi, "diabetes": diabetes,
            }
            st.session_state.page = "loading"
            st.rerun()

    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="footer">
            <strong>⚠️ Disclaimer:</strong> This tool is for research and educational purposes only.
            It is not a substitute for professional medical diagnosis.<br>
            Built with Streamlit • TensorFlow • EfficientNetB3 • Grad-CAM
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 – LOADING / PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "loading":

    render_step_bar(2)

    # ── Animated loading UI ──────────────────────────────────────────────────────
    loading_placeholder = st.empty()

    loading_steps = [
        "🔬 Preprocessing retinal images …",
        "🧠 Running deep-learning inference …",
        "🔥 Generating Grad-CAM heatmaps …",
        "📊 Classifying cardiovascular risk …",
        "✅ Finalizing results …",
    ]

    loading_placeholder.markdown(
        f"""
        <div class="loading-container">
            <div class="pulse-wrapper">
                <div class="ring"></div>
                <div class="ring"></div>
                <div class="ring"></div>
                <div class="icon">👁️</div>
            </div>
            <div class="loading-title">Analyzing Retinal Images</div>
            <div class="loading-sub">{loading_steps[0]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    progress_bar = st.progress(0, text="Starting analysis …")

    # ── Actual processing ────────────────────────────────────────────────────────
    file_data = st.session_state.uploaded_data
    clin = st.session_state.clinical
    clinical_data = prepare_clinical_data(clin["bp"], clin["cholesterol"], clin["bmi"], clin["diabetes"])

    total = len(file_data)
    results = []

    # Calculate minimum per-image sleep to guarantee 3-5 s total visible loading
    MIN_TOTAL_SECONDS = 4.0
    per_image_delay = max(0, MIN_TOTAL_SECONDS / max(total, 1))

    for i, fd in enumerate(file_data):
        step_idx = min(i, len(loading_steps) - 1)

        # Update loading animation text
        loading_placeholder.markdown(
            f"""
            <div class="loading-container">
                <div class="pulse-wrapper">
                    <div class="ring"></div>
                    <div class="ring"></div>
                    <div class="ring"></div>
                    <div class="icon">👁️</div>
                </div>
                <div class="loading-title">Analyzing Retinal Images</div>
                <div class="loading-sub">{loading_steps[step_idx]}</div>
                <div style="color:#667eea; margin-top:1rem; font-weight:600;">
                    Image {i + 1} / {total}  —  {fd['name']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        progress_bar.progress((i) / total, text=f"Processing image {i + 1} of {total} …")

        step_start = time.time()

        # Build a file-like object from stored bytes
        import io as _io
        file_obj = _io.BytesIO(fd["bytes"])
        file_obj.name = fd["name"]

        result_entry = {"name": fd["name"], "error": None}

        try:
            image_array, original_image = preprocess_image(file_obj)
        except (ValueError, Exception) as e:
            result_entry["error"] = f"Could not process image: {e}"
            results.append(result_entry)
            continue

        # Validation
        if not is_retinal_image(original_image):
            result_entry["error"] = "Invalid image — not recognized as a retinal fundus photograph."
            result_entry["original_image"] = original_image
            results.append(result_entry)
            continue

        # Prediction
        probability = predict(model, image_array, clinical_data)
        risk = interpret_risk(probability)

        # Grad-CAM
        try:
            heatmap = generate_gradcam(model, image_array, clinical_data)
            gradcam_image = overlay_heatmap(heatmap, original_image)
        except Exception:
            gradcam_image = None

        result_entry.update({
            "original_image": original_image,
            "probability": probability,
            "risk": risk,
            "gradcam_image": gradcam_image,
        })
        results.append(result_entry)

        # Ensure minimum visible loading time
        elapsed = time.time() - step_start
        remaining_delay = per_image_delay - elapsed
        if remaining_delay > 0:
            time.sleep(remaining_delay)

    progress_bar.progress(1.0, text="✅ Analysis complete!")
    time.sleep(0.6)

    # Store results and navigate
    st.session_state.results = results
    st.session_state.page = "results"
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 – RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "results":

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>👁️ RetinaRisk — Results</h1>
        <p>Analysis Complete • Images Classified by Cardiovascular Risk</p>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    render_step_bar(3)

    results = st.session_state.results

    # Navigate back button
    if st.button("← Back to Upload", type="secondary"):
        st.session_state.page = "upload"
        st.session_state.results = []
        st.session_state.uploaded_data = []
        st.rerun()

    # ── Separate results into risk groups ────────────────────────────────────────
    high_risk = []   # percent >= 60 (High / Critical)
    low_risk = []    # percent < 60  (Very Low / Low / Moderate)
    errors = []

    for r in results:
        if r.get("error"):
            errors.append(r)
        elif r["risk"]["percent"] >= 60:
            high_risk.append(r)
        else:
            low_risk.append(r)

    # ── Summary metrics ──────────────────────────────────────────────────────────
    total_valid = len(high_risk) + len(low_risk)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Images", len(results))
    col_m2.metric("Valid Scans", total_valid)
    col_m3.metric("🔴 High Risk", len(high_risk))
    col_m4.metric("🟢 Low Risk", len(low_risk))

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Errors / Rejected images ─────────────────────────────────────────────────
    if errors:
        with st.expander(f"⚠️ Rejected / Failed Images ({len(errors)})", expanded=False):
            for r in errors:
                st.warning(f"**{r['name']}** — {r['error']}")
                if r.get("original_image") is not None:
                    st.image(r["original_image"], width=200, caption=r["name"])

    # ── HIGH RISK GROUP ──────────────────────────────────────────────────────────
    if high_risk:
        st.markdown(
            '<div class="risk-group-header high">🔴 High Risk Images</div>',
            unsafe_allow_html=True,
        )

        for idx, r in enumerate(high_risk):
            _render_result_card(r, f"high_{idx}")

    # ── LOW RISK GROUP ───────────────────────────────────────────────────────────
    if low_risk:
        st.markdown(
            '<div class="risk-group-header low">🟢 Low Risk Images</div>',
            unsafe_allow_html=True,
        )

        for idx, r in enumerate(low_risk):
            _render_result_card(r, f"low_{idx}")

    # ── Clinical Summary ─────────────────────────────────────────────────────────
    if "clinical" in st.session_state:
        clin = st.session_state.clinical
        with st.expander("📊 Clinical Input Used for Prediction(s)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Blood Pressure", f"{clin['bp']} mmHg")
            c2.metric("Cholesterol", f"{clin['cholesterol']} mg/dL")
            c3.metric("BMI", f"{clin['bmi']:.1f}")
            c4.metric("Diabetes", "Yes" if clin["diabetes"] else "No")

    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="footer">
            <strong>⚠️ Disclaimer:</strong> This tool is for research and educational purposes only.
            It is not a substitute for professional medical diagnosis.<br>
            Built with Streamlit • TensorFlow • EfficientNetB3 • Grad-CAM
        </div>
        """,
        unsafe_allow_html=True,
    )

