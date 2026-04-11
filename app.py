"""
app.py
Main Streamlit application for Retinal Cardiovascular Risk Prediction.
Provides image upload, clinical data inputs, risk prediction, Grad-CAM visualization,
and multi-image batch validation support.
"""

import io
import streamlit as st
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
</style>
""", unsafe_allow_html=True)


# ─── Helper UI Components ───────────────────────────────────────────────────────
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


# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>👁️ RetinaRisk</h1>
    <p>AI-Powered Cardiovascular Risk Prediction from Retinal Fundus Images</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────────
try:
    with st.spinner("🔧 Loading AI model — this may take a moment on first run..."):
        model = load_model()
except (FileNotFoundError, RuntimeError) as e:
    st.error(f"❌ **Model Loading Error**\n\n{e}")
    st.stop()


# ─── Sidebar: Clinical Data Inputs ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Clinical Data")
    st.caption("Adjust the clinical parameters below. These apply to all uploaded images.")

    st.markdown("---")

    bp = st.slider(
        "Systolic Blood Pressure (mmHg)",
        min_value=80, max_value=200, value=120, step=1,
        help="Normal: 90-120 mmHg. Hypertension: >140 mmHg."
    )

    cholesterol = st.slider(
        "Total Cholesterol (mg/dL)",
        min_value=100, max_value=400, value=200, step=5,
        help="Desirable: <200 mg/dL. High: >240 mg/dL."
    )

    bmi = st.slider(
        "Body Mass Index (BMI)",
        min_value=15.0, max_value=45.0, value=25.0, step=0.5,
        help="Normal: 18.5-24.9. Overweight: 25-29.9. Obese: ≥30."
    )

    diabetes = st.selectbox(
        "Diabetes Status",
        options=[0, 1],
        format_func=lambda x: "No Diabetes" if x == 0 else "Diabetic",
        help="Select current diabetes status."
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.8rem;'>"
        "These values are combined with retinal analysis for a multi-modal prediction."
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Image Upload (Batch Support) ─────────────────────────────────────────────────
st.markdown("### 📤 Upload Retinal Fundus Image(s)")

uploaded_files = st.file_uploader(
    "Choose retinal fundus images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload high-quality retinal fundus photographs. Non-retinal images will be rejected.",
    label_visibility="collapsed",
)

if uploaded_files:
    clinical_data = prepare_clinical_data(bp, cholesterol, bmi, diabetes)

    for idx, uploaded_file in enumerate(uploaded_files):
        # UI Container for each image
        with st.container():
            st.markdown(f"#### 📄 Image: `{uploaded_file.name}`")
            
            # ── Preprocess Image ─────────────────────────────────────────────────────
            try:
                # Seek to 0 in case the file was read in a previous run loop (Streamlit handles this mostly)
                uploaded_file.seek(0)
                image_array, original_image = preprocess_image(uploaded_file)
            except ValueError as e:
                st.error(f"❌ **Error processing file**: {e}")
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                continue
            except Exception as e:
                st.error(f"❌ **Unexpected Error**: Could not process the uploaded image.\n\n{e}")
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                continue

            # ── Validation ───────────────────────────────────────────────────────────
            if not is_retinal_image(original_image):
                st.image(original_image, use_container_width=False, width=300, caption="Uploaded Image")
                # Warning banner and hard block for invalid input
                st.warning("⚠️ **Validation Failed**")
                st.error("❌ **Invalid image: Please upload a retinal fundus image.**")
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                continue

            # ── Run Prediction ───────────────────────────────────────────────────────
            with st.spinner("🔍 Analyzing retinal features..."):
                probability = predict(model, image_array, clinical_data)
                risk = interpret_risk(probability)

                # Generate Grad-CAM
                try:
                    heatmap = generate_gradcam(model, image_array, clinical_data)
                    gradcam_image = overlay_heatmap(heatmap, original_image)
                except Exception as e:
                    gradcam_image = None
                    st.warning(f"⚠️ Grad-CAM generation failed: {e}")

            # ── Results UI ───────────────────────────────────────────────────────────
            
            # Dynamic Risk Label
            display_risk_label(risk["label"], risk["emoji"], risk["color"])

            # Large Percentage Score Card
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-value" style="color: {risk['color']}">{risk['percent']:.2f}%</div>
                    <div class="score-subtitle">Cardiovascular Risk Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence / Risk Progress Bar (Streamlit progress expects 0.0-1.0 float)
            st.progress(float(probability), text=f"Risk Measure: {risk['percent']:.2f}%")

            # Text Interpretation
            st.markdown(
                f'<div class="interpretation">💡 {risk["interpretation"]}</div>',
                unsafe_allow_html=True,
            )

            # ── Image Display: Original | Grad-CAM ───────────────────────────────────
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### 🖼️ Original Retinal Scan")
                st.image(original_image, use_container_width=True)

            with col2:
                st.markdown("##### 🔥 Grad-CAM Heatmap")
                if gradcam_image is not None:
                    st.image(gradcam_image, use_container_width=True)

                    # Download button for Grad-CAM
                    gradcam_pil = Image.fromarray(gradcam_image)
                    buf = io.BytesIO()
                    gradcam_pil.save(buf, format="PNG")
                    buf.seek(0)

                    st.download_button(
                        label="⬇️ Download Grad-CAM Overlay",
                        data=buf,
                        file_name=f"gradcam_{uploaded_file.name}",
                        mime="image/png",
                        use_container_width=True,
                        key=f"download_{idx}"
                    )
                else:
                    st.info("Grad-CAM visualization unavailable for this image.")
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Clinical Summary ─────────────────────────────────────────────────────
    with st.expander("📊 Clinical Input Used for Prediction(s)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Blood Pressure", f"{bp} mmHg")
        c2.metric("Cholesterol", f"{cholesterol} mg/dL")
        c3.metric("BMI", f"{bmi:.1f}")
        c4.metric("Diabetes", "Yes" if diabetes else "No")

else:
    # ── Placeholder when no image is uploaded ────────────────────────────────
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


# ─── Footer ─────────────────────────────────────────────────────────────────────
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
