# RetinaRisk — Complete Project Explanation

## 1. Project Overview

**RetinaRisk** is an AI-powered web application that predicts cardiovascular (heart disease) risk by analyzing retinal fundus images — photographs of the back of the eye. The system uses a deep learning model based on **EfficientNetB3** combined with clinical patient data to produce a risk score between 0 and 1.

The core idea behind this project is rooted in medical research showing that the blood vessels in the retina mirror the health of blood vessels throughout the body. Changes in retinal vasculature (narrowing, widening, micro-hemorrhages) can be early indicators of cardiovascular disease, even before traditional symptoms appear.

The application uses a **Flask** backend for API-based inference and a **pure HTML/CSS/JS** frontend for a clean, responsive, multi-page user experience.

---

## 2. How It Works (End-to-End Flow)

```
User uploads retinal image(s) + sets clinical data on Upload Page
        │
        ▼
┌─────────────────────────┐
│  Click "Analyze"        │  ← Sends images + clinical data as POST /api/analyze
│  → Show Loading Screen  │     Animated pulsing icon, progress bar, step messages
│    (minimum ~3.5 secs)  │     (3.5s minimum visible loading time)
└────────────┬────────────┘
             │
     ┌───────┴───── Server-side Processing ─────┐
     │                                           │
     │  FOR EACH uploaded image:                 │
     │                                           │
     │  ┌─────────────────────────┐              │
     │  │  Image Validation       │  ← HSV color profiling + Hough Circle detection
     │  └────────────┬────────────┘              │
     │               │ Valid                     │
     │               ▼                           │
     │  ┌─────────────────────────┐              │
     │  │  Image Preprocessing    │  ← Resizes to 224×224, float32 array
     │  └────────────┬────────────┘              │
     │               │                           │
     │               ▼                           │
     │  ┌─────────────────────┐  ┌──────────────┐│
     │  │  EfficientNetB3     │  │ Clinical Data ││
     │  │  → 1536 features    │  │ → 32 features ││
     │  └──────────┬──────────┘  └───────┬──────┘│
     │             └──────┬──────────────┘       │
     │                    │ Concatenation         │
     │                    ▼                       │
     │          ┌───────────────────┐             │
     │          │  Dense Layers     │             │
     │          │  128→64→32→1      │             │
     │          │  (sigmoid)        │             │
     │          └─────────┬─────────┘             │
     │                    │                       │
     │                    ▼                       │
     │          ┌───────────────────┐             │
     │          │  Risk Probability │             │
     │          │  + Grad-CAM       │             │
     │          └───────────────────┘             │
     │                                           │
     └───────────────────────────────────────────┘
             │
             ▼  JSON Response (results + base64 images)
┌─────────────────────────┐
│  Results Page            │
│  ┌───────────────────┐  │
│  │ 🔴 High Risk Group│  │  ← Images with risk score ≥ 60%
│  │  - Result cards   │  │
│  │  - Heatmap button │  │  ← Click to open Grad-CAM modal
│  └───────────────────┘  │
│  ┌───────────────────┐  │
│  │ 🟢 Low Risk Group │  │  ← Images with risk score < 60%
│  │  - Result cards   │  │
│  │  - Heatmap button │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

---

## 3. Project File Structure

| File | Purpose |
|------|---------|
| `server.py` | Flask backend — serves static frontend, exposes `/api/analyze` endpoint |
| `static/index.html` | Single-page HTML frontend with 3 views (Upload, Loading, Results) |
| `static/style.css` | Dark-themed CSS with animations, responsive layout, glass morphism |
| `static/script.js` | Frontend JavaScript — file uploads, API calls, result rendering, heatmap modal |
| `model_loader.py` | Loads the pre-trained Keras model with architecture reconstruction |
| `predictor.py` | Image preprocessing, validation, and risk prediction logic |
| `gradcam.py` | Grad-CAM heatmap generation for model explainability |
| `requirements.txt` | Python package dependencies |
| `retina_heart_model.h5` | Pre-trained model weights (~96 MB) |
| `retina_(1)_(1).ipynb` | Jupyter notebook used for model training and experimentation |
| `kaggle (1).json` | Kaggle API credentials for dataset downloading |
| `app.py` | Legacy Streamlit application (alternative entry point) |

---

## 4. Detailed File Explanations

### 4.1 `server.py` — Flask Backend

This is the **Flask web server** that serves both the static frontend and the prediction API. It handles:

- **Static File Serving**: Serves `static/index.html`, `style.css`, and `script.js` from the `static/` directory.
- **Model Loading at Startup**: Loads the EfficientNetB3 model once when the server starts (no per-request loading overhead).
- **`/api/analyze` Endpoint (POST)**: Accepts multipart form data containing:
  - `images`: Multiple image files
  - `bp`, `cholesterol`, `bmi`, `diabetes`: Clinical parameters
- **Per-Image Processing**: For each uploaded image:
  1. Decodes and validates the image
  2. Checks if it's a retinal fundus image
  3. Runs model inference
  4. Generates Grad-CAM heatmap
  5. Encodes original image and heatmap as base64 PNG strings
- **JSON Response**: Returns a JSON array of results, each containing the risk score, risk label, original image (base64), and Grad-CAM overlay (base64).
- **Helper Function `numpy_to_base64()`**: Converts NumPy image arrays to base64-encoded PNG strings for transmission to the frontend.

### 4.2 `static/index.html` — Frontend HTML

The single-page HTML file contains three `<section>` elements representing the three views:

- **Page 1 — Upload**: Contains the step bar, header, clinical data panel (left sidebar), and drag-and-drop upload area (right). Includes a preview strip for selected images and the "Analyze All Images" button.
- **Page 2 — Loading**: An animated processing screen with a pulsing eye icon, progress bar, and step-by-step status text.
- **Page 3 — Results**: Displays summary metrics (total images, valid scans, high/low risk counts), error section for rejected images, risk-grouped result cards, clinical summary, and footer.
- **Heatmap Modal**: A full-screen overlay with original and Grad-CAM images side-by-side, and a download button.

### 4.3 `static/style.css` — Stylesheet

A comprehensive dark-themed stylesheet featuring:
- **CSS Variables**: Central design tokens for colors, borders, radii, and transitions
- **Page Transitions**: Fade-in animations when switching views
- **Step Bar**: Visual 3-step progress indicator with done/active/inactive states
- **Loading Animations**: Pulsing rings (`@keyframes pulse-ring`), bouncing icon, floating text
- **Result Cards**: Hover-lift effect, risk badges, progress bars
- **Modal**: Backdrop blur, scale-in animation, responsive image grid
- **Responsive Design**: Grid layout adapts to mobile screens via media queries

### 4.4 `static/script.js` — Frontend Logic

The JavaScript handles all client-side interactions:
- **File Management**: Drag-and-drop, click-to-browse, duplicate detection, thumbnail previews with remove buttons
- **Page Navigation**: `showPage(id)` function that toggles CSS class `active` on page sections
- **API Communication**: Sends multipart `FormData` via `fetch()` to `/api/analyze`
- **Loading Animation**: Simulates fake progress (minimum 3.5s display), then transitions to results
- **Results Rendering**: Separates results into high risk (≥60%) and low risk (<60%), generates metric cards and result card HTML
- **Heatmap Modal**: Opens on button click with original + Grad-CAM side-by-side, supports download and Escape key to close
- **XSS Prevention**: `esc()` function sanitizes user-provided strings before HTML insertion

### 4.5 `model_loader.py` — Model Loading

This module handles loading the deep learning model:

- **Architecture Reconstruction**: Instead of loading the full saved model (which can cause Keras version incompatibility issues), it:
  1. Rebuilds the exact model architecture programmatically
  2. Creates the image input branch (EfficientNetB3 backbone)
  3. Creates the clinical data input branch (Dense layers)
  4. Concatenates both branches
  5. Adds final classification layers
  6. Loads only the weights from the `.h5` file using `by_name=True, skip_mismatch=True`
- **Note**: The `@st.cache_resource` decorator in this file is a Streamlit-specific caching mechanism. In the Flask backend (`server.py`), the model is loaded once at module level instead, achieving the same single-load behavior.
- **Error Handling**: Raises clear exceptions if the model file is missing or weights loading fails.

**Model Architecture Details:**
```
Image Input (224×224×3)
    → EfficientNetB3 (no top, no pretrained weights at load)
    → GlobalAveragePooling2D
    → BatchNormalization
    → Dropout(0.3)
                            ─┐
                             │ Concatenate
Clinical Input (4 features)  │
    → Dense(64, relu)       ─┘
    → BatchNormalization
    → Dense(32, relu)

Combined
    → Dense(128, relu) → BatchNorm → Dropout(0.3)
    → Dense(64, relu) → Dropout(0.2)
    → Dense(32, relu)
    → Dense(1, sigmoid) → Output probability
```

### 4.6 `predictor.py` — Prediction Logic

This module contains four key functions:

#### `is_retinal_image(image_rgb)` — Image Validation
Determines if an uploaded image is genuinely a retinal fundus photograph using a two-stage approach:
1. **HSV Color Analysis**: Converts the image to HSV color space and checks:
   - Whether the image has enough non-black content (>10% of pixels)
   - Whether average saturation is above 25 (filters out grayscale images)
   - Whether >35% of content pixels fall in the red/orange hue range (H: 0–45 or 150–179), characteristic of retinal images
2. **Hough Circle Detection (Fallback)**: If color analysis is inconclusive, uses OpenCV's Hough Circle Transform to detect the circular fundus mask typical of retinal photographs.

#### `preprocess_image(uploaded_file)` — Image Preprocessing
- Reads raw file bytes and decodes using OpenCV
- Converts BGR → RGB
- Resizes to 224×224 pixels
- Converts to float32 and adds batch dimension
- Returns both the preprocessed array (for model) and the original RGB image (for display)

#### `prepare_clinical_data(bp, cholesterol, bmi, diabetes)` — Clinical Data Packaging
- Takes the four clinical parameters and packages them into a NumPy array of shape `(1, 4)` with float32 dtype.

#### `predict(model, image_array, clinical_data)` — Model Inference
- Runs inference using `model([image_array, clinical_data], training=False)` instead of `model.predict()` to avoid threading issues on some platforms.
- Returns a single float probability between 0 and 1.

#### `interpret_risk(probability)` — Risk Interpretation
Maps the raw probability to a human-readable assessment:

| Score Range | Label | Color | Emoji |
|-------------|-------|-------|-------|
| 0–20% | Very Low Risk | 🟢 Green | 🟢 |
| 21–40% | Low Risk | 🟢 Green | 🟢 |
| 41–60% | Moderate Risk | 🟡 Yellow | 🟡 |
| 61–80% | High Risk | 🔴 Red | 🔴 |
| 81–100% | Critical Risk | 🔴 Red | 🔴 |

Each tier includes a plain-text interpretation explaining the clinical significance.

### 4.7 `gradcam.py` — Grad-CAM Explainability

This module implements **Gradient-weighted Class Activation Mapping (Grad-CAM)**, a technique that provides visual explanations for which regions of the retinal image the model "looks at" when making its prediction.

#### `find_last_conv_layer(model)`
- Iterates through the model layers in reverse to find the last Conv2D layer (deepest convolutional layer in EfficientNetB3).
- This layer's feature maps contain the most semantically rich spatial information.

#### `generate_gradcam(model, image_array, clinical_data)`
The core Grad-CAM algorithm:
1. Creates a sub-model that outputs both the last conv layer's activations and the final prediction
2. Uses `tf.GradientTape` to compute gradients of the prediction with respect to the conv layer output
3. Computes global average pooling of gradients to get channel importance weights
4. Creates a weighted sum of feature maps
5. Applies ReLU (keeps only positive influences)
6. Normalizes to [0, 1] range
7. Resizes to 224×224

#### `overlay_heatmap(heatmap, original_image)`
- Converts the heatmap to a color-mapped image using OpenCV's JET colormap (blue=cold/unimportant, red=hot/important)
- Blends it with the original image using configurable alpha (default 0.4)
- Returns the superimposed RGB image

---

## 5. Technology Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.9+ | Core programming language |
| **Flask** | ≥3.0.0 | Lightweight web framework (backend API server) |
| **Streamlit** | ≥1.30.0 | Rapid web application framework (Legacy UI) |
| **HTML5/CSS3/JS** | — | Frontend user interface (single-page application) |
| **TensorFlow/Keras** | ≥2.15.0 | Deep learning model and inference |
| **OpenCV** | ≥4.8.0 (headless) | Image processing, Grad-CAM overlay |
| **NumPy** | ≥1.24.0 | Numerical computations |
| **Pillow** | ≥10.0.0 | Image format conversion for base64 encoding |

---

## 6. Deep Learning Model Details

### Architecture: EfficientNetB3
- **EfficientNet** is a family of convolutional neural networks that uniformly scale depth, width, and resolution using a compound scaling method.
- **B3** is the third variant, offering a good balance between accuracy and computational cost.
- Originally pre-trained on **ImageNet** (1.4M images, 1000 classes), the network learned general visual features that transfer well to medical imaging tasks.

### Multi-Input Design
The model is **multi-modal** — it doesn't rely on the retinal image alone. It combines:
1. **Visual features** from the retinal image (1536-dimensional vector from EfficientNetB3's global average pooling)
2. **Clinical features** (4 values: blood pressure, cholesterol, BMI, diabetes status)

This multi-modal approach improves prediction accuracy by incorporating known cardiovascular risk factors alongside the visual analysis.

### Training
The model was trained in the Jupyter notebook (`retina_(1)_(1).ipynb`) using a dataset likely sourced from Kaggle (evidenced by the `kaggle (1).json` credentials file). The training process involved:
- Dataset distribution analysis
- Image preprocessing
- Model architecture definition
- Training with loss and accuracy monitoring
- Evaluation via confusion matrix and test predictions

---

## 7. Key Design Decisions

1. **Flask over Streamlit**: The project was migrated from Streamlit to Flask + HTML/CSS/JS to gain full control over the UI/UX flow, enabling multi-page navigation (Upload → Loading → Results), custom animations, and a proper modal for heatmap viewing.

2. **Client-server architecture**: The Flask backend exposes a single `/api/analyze` endpoint. The frontend communicates via `fetch()` with `FormData`, and results are returned as JSON with base64-encoded images.

3. **Minimum loading time**: The frontend enforces a minimum ~3.5 second loading animation to give users a sense of processing, even if the backend responds faster.

4. **Risk classification on the results page**: Images are automatically sorted into High Risk (≥60%) and Low Risk (<60%) groups for quick triage.

5. **On-demand heatmap viewing**: Grad-CAM heatmaps are not shown by default — users click a button to open them in a full-screen modal with download capability, keeping the results page clean.

6. **Weight-only loading** (`model_loader.py`): The model architecture is rebuilt in code rather than loading the full saved model. This avoids Keras serialization version mismatches that commonly occur between different TensorFlow versions.

7. **Retinal image validation** (`predictor.py`): A multi-stage validation pipeline ensures only valid retinal images are processed, preventing meaningless predictions on random photos.

8. **`model()` vs `model.predict()`**: Direct model calling is used instead of `.predict()` to avoid threading bugs on macOS and other platforms.

9. **Grad-CAM for trust**: By showing which retinal regions influenced the prediction, the system builds transparency and trust — critical for medical AI applications.

---

## 8. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure retina_heart_model.h5 is in the project root

# 3. Launch the app (Choose one)

# Option A: Flask Web App (Recommended)
python server.py
# Opens at http://127.0.0.1:5000

# Option B: Streamlit App
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 9. Disclaimer

This tool is for **research and educational purposes only**. It is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for medical advice.
