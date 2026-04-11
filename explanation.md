# RetinaRisk — Complete Project Explanation

## 1. Project Overview

**RetinaRisk** is an AI-powered web application that predicts cardiovascular (heart disease) risk by analyzing retinal fundus images — photographs of the back of the eye. The system uses a deep learning model based on **EfficientNetB3** combined with clinical patient data to produce a risk score between 0 and 1.

The core idea behind this project is rooted in medical research showing that the blood vessels in the retina mirror the health of blood vessels throughout the body. Changes in retinal vasculature (narrowing, widening, micro-hemorrhages) can be early indicators of cardiovascular disease, even before traditional symptoms appear.

---

## 2. How It Works (End-to-End Flow)

```
User uploads retinal image + enters clinical data
        │
        ▼
┌─────────────────────────┐
│  Image Validation       │  ← Checks if the image is actually a retinal fundus photo
│  (HSV color analysis    │     using color profiling and Hough Circle detection
│   + Hough Circles)      │
└────────────┬────────────┘
             │ Valid
             ▼
┌─────────────────────────┐
│  Image Preprocessing    │  ← Resizes to 224×224, converts to float32 array
│  (OpenCV + NumPy)       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐     ┌──────────────────────┐
│  EfficientNetB3         │     │  Clinical Data       │
│  (Feature Extraction)   │     │  (BP, Cholesterol,   │
│  Image → 1536 features  │     │   BMI, Diabetes)     │
└────────────┬────────────┘     │  → Dense layers      │
             │                  │  → 32 features       │
             │                  └──────────┬───────────┘
             │                             │
             └──────────┬──────────────────┘
                        │ Concatenation
                        ▼
              ┌───────────────────┐
              │  Fully Connected  │
              │  Layers (128→64→  │
              │  32→1 sigmoid)    │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  Risk Probability │  ← Value between 0.0 and 1.0
              │  + Grad-CAM       │  ← Heatmap showing which retinal regions matter
              └───────────────────┘
```

---

## 3. Project File Structure

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit web application — UI, layout, user interactions |
| `model_loader.py` | Loads the pre-trained Keras model with caching |
| `predictor.py` | Image preprocessing, validation, and risk prediction logic |
| `gradcam.py` | Grad-CAM heatmap generation for model explainability |
| `requirements.txt` | Python package dependencies |
| `retina_heart_model.h5` | Pre-trained model weights (~96 MB) |
| `retina_(1)_(1).ipynb` | Jupyter notebook used for model training and experimentation |
| `kaggle (1).json` | Kaggle API credentials for dataset downloading |

---

## 4. Detailed File Explanations

### 4.1 `app.py` — Main Application (344 lines)

This is the **Streamlit web application** that serves as the user-facing interface. It handles:

- **Page Configuration**: Sets the page title, icon, and layout to wide mode with an expanded sidebar.
- **Custom CSS Styling**: Injects premium dark-themed CSS with:
  - Google Fonts (Inter)
  - Gradient backgrounds and glassmorphism-style cards
  - Custom score cards, dividers, and footer styling
- **Sidebar — Clinical Data Inputs**: Provides sliders and dropdowns for:
  - Systolic Blood Pressure (80–200 mmHg)
  - Total Cholesterol (100–400 mg/dL)
  - BMI (15.0–45.0)
  - Diabetes Status (Yes/No)
- **Image Upload**: Supports batch upload of multiple retinal images (JPG/PNG).
- **Per-Image Processing Loop**:
  1. Preprocesses each uploaded image
  2. Validates it as a retinal fundus image
  3. Runs the prediction model
  4. Generates Grad-CAM heatmap
  5. Displays results: risk label, percentage score, progress bar, interpretation text, original image, and Grad-CAM overlay
- **Download Feature**: Users can download the Grad-CAM overlay as a PNG file.
- **Clinical Summary**: An expandable section showing the clinical parameters used.

### 4.2 `model_loader.py` — Model Loading (76 lines)

This module handles loading the deep learning model:

- **Architecture Reconstruction**: Instead of loading the full saved model (which can cause Keras version incompatibility issues), it:
  1. Rebuilds the exact model architecture programmatically
  2. Creates the image input branch (EfficientNetB3 backbone)
  3. Creates the clinical data input branch (Dense layers)
  4. Concatenates both branches
  5. Adds final classification layers
  6. Loads only the weights from the `.h5` file using `by_name=True, skip_mismatch=True`
- **Caching**: Uses `@st.cache_resource` to ensure the model is loaded only once per session, dramatically improving performance on subsequent requests.
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

### 4.3 `predictor.py` — Prediction Logic (183 lines)

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

### 4.4 `gradcam.py` — Grad-CAM Explainability (122 lines)

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
| **Streamlit** | ≥1.30.0 | Web application framework |
| **TensorFlow/Keras** | ≥2.15.0 | Deep learning model and inference |
| **OpenCV** | ≥4.8.0 (headless) | Image processing, Grad-CAM overlay |
| **NumPy** | ≥1.24.0 | Numerical computations |
| **Pillow** | ≥10.0.0 | Image format conversion for download |

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

1. **Weight-only loading** (`model_loader.py`): The model architecture is rebuilt in code rather than loading the full saved model. This avoids Keras serialization version mismatches that commonly occur between different TensorFlow versions.

2. **Retinal image validation** (`predictor.py`): A multi-stage validation pipeline ensures only valid retinal images are processed, preventing meaningless predictions on random photos.

3. **`model()` vs `model.predict()`**: Direct model calling is used instead of `.predict()` to avoid threading bugs on macOS and other platforms with Streamlit.

4. **Streamlit caching**: `@st.cache_resource` ensures the ~96 MB model is loaded into memory only once, making subsequent image analyses near-instant.

5. **Grad-CAM for trust**: By showing which retinal regions influenced the prediction, the system builds transparency and trust — critical for medical AI applications.

---

## 8. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure retina_heart_model.h5 is in the project root

# 3. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 9. Disclaimer

This tool is for **research and educational purposes only**. It is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for medical advice.
