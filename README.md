# 👁️ RetinaRisk — Cardiovascular Risk Prediction from Retinal Images

An AI-powered web application that predicts cardiovascular risk from retinal fundus images using a pre-trained **EfficientNetB3** deep learning model with **Grad-CAM** explainability.

---

## 🎯 Features

- **Retinal Image Analysis** — Upload fundus photographs and get instant cardiovascular risk scores
- **Multi-Modal Prediction** — Combines retinal image features with clinical data (BP, cholesterol, BMI, diabetes)
- **Grad-CAM Visualization** — View which regions of the retina the model focuses on via an on-demand modal
- **Downloadable Heatmaps** — Export the Grad-CAM overlay as a PNG
- **Batch Upload** — Analyze multiple retinal images at once with drag-and-drop support
- **Image Validation** — Automatically rejects non-retinal images
- **Risk Classification** — Results page separates images into **High Risk** and **Low Risk** groups
- **Multi-Page UX** — Clean 3-step flow: Upload → Processing (animated loading) → Results
- **Premium Dark UI** — Modern, responsive HTML/CSS/JS interface with animations and glass morphism

---

## 🛠️ Prerequisites

| Requirement | Version |
|-------------|---------|
| **Python** | 3.9 or higher |
| **pip** | Latest recommended |
| **OS** | Windows, macOS, or Linux |
| **RAM** | 4 GB minimum (8 GB recommended for smooth model loading) |
| **Disk Space** | ~200 MB (model file is ~96 MB) |

You also need the pre-trained model file **`retina_heart_model.h5`** in the project root directory. This file contains the trained weights for the EfficientNetB3 model.

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| `flask` | ≥ 3.0.0 | Lightweight Python web framework (backend API server) |
| `tensorflow` | ≥ 2.15.0 | Deep learning model & inference |
| `opencv-python-headless` | ≥ 4.8.0 | Image processing & Grad-CAM overlay |
| `numpy` | ≥ 1.24.0 | Numerical computations |
| `Pillow` | ≥ 10.0.0 | Image format conversion for base64 encoding |

---

## 🚀 How to Run

### Step 1: Clone or Navigate to the Project

```bash
cd /path/to/retina
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow installation may take a few minutes and requires ~500 MB of disk space. If you have a compatible GPU, TensorFlow will automatically use it for faster inference.

### Step 4: Verify the Model File

Make sure `retina_heart_model.h5` exists in the project root:

```
retina/
├── server.py
├── retina_heart_model.h5   ← This file must be present
├── ...
```

### Step 5: Launch the Application

```bash
python server.py
```

The app will be available in your browser at **`http://127.0.0.1:5000`**.

> **First launch** may take 30–60 seconds while the model loads into memory. The model is loaded once at server startup; subsequent analyses are near-instant.

---

## 📁 Project Structure

```
retina/
├── server.py                # Flask backend (API + static file serving)
├── static/
│   ├── index.html           # Single-page frontend (3-view: Upload → Loading → Results)
│   ├── style.css            # Dark-themed CSS with animations
│   └── script.js            # Frontend logic (uploads, API calls, rendering)
├── model_loader.py          # Model architecture reconstruction & weight loading
├── predictor.py             # Image preprocessing, validation & prediction logic
├── gradcam.py               # Grad-CAM heatmap generation
├── app.py                   # Legacy Streamlit application (no longer primary)
├── requirements.txt         # Python dependencies
├── retina_heart_model.h5    # Pre-trained model weights (~96 MB)
├── retina_(1)_(1).ipynb     # Jupyter notebook (model training & experimentation)
├── kaggle (1).json          # Kaggle API credentials for dataset access
├── documentation.md         # Detailed project documentation
├── explanation.md           # Technical project explanation
└── README.md                # This file
```

---

## 📊 Model Architecture

| Component          | Detail                                      |
|--------------------|---------------------------------------------|
| **Backbone**       | EfficientNetB3 (ImageNet pre-trained)       |
| **Image Input**    | 224 × 224 × 3 RGB                           |
| **Clinical Input** | 4 features (BP, Cholesterol, BMI, Diabetes) |
| **Output**         | Single sigmoid neuron (0–1 probability)     |
| **Threshold**      | 0.5 (≥ 0.5 = High Risk)                    |

---

## 🖥️ Usage

1. Open the app by running `python server.py` and navigating to `http://127.0.0.1:5000`
2. **Upload Page (Step 1):**
   - Adjust clinical parameters in the left panel (or leave defaults):
     - Systolic Blood Pressure (80–200 mmHg)
     - Total Cholesterol (100–400 mg/dL)
     - BMI (15.0–45.0)
     - Diabetes Status (Yes / No)
   - Drag-and-drop or click to upload one or more retinal fundus images (JPG / PNG)
   - Click **"🚀 Analyze All Images"**
3. **Processing (Step 2):**
   - An animated loading screen appears with a pulsing icon and progress bar
   - Step-by-step status messages are shown (minimum ~3.5 seconds display time)
4. **Results Page (Step 3):**
   - Images are classified into **🔴 High Risk** and **🟢 Low Risk** groups
   - Each result card shows:
     - **Risk Score** — Percentage from 0% to 100%
     - **Risk Label** — Color-coded risk tier (Very Low → Critical)
     - **Progress Bar** — Visual risk indicator
     - **Interpretation** — Plain-text explanation
     - **Retinal Scan Thumbnail** — The uploaded image
   - Click **"🔥 View Grad-CAM Heatmap"** to open a modal with the original and heatmap overlay side-by-side
   - Download the Grad-CAM overlay from the modal
   - Click **"← Back to Upload"** to analyze more images

### Output Interpretation

| Score Range  | Label          | Interpretation                                                   |
|-------------|----------------|------------------------------------------------------------------|
| 0% – 20%   | Very Low Risk  | Retinal vasculature appears exceptionally healthy                |
| 21% – 40%  | Low Risk       | Healthy retina with minimal risk indicators                      |
| 41% – 60%  | Moderate Risk  | Borderline indicators — further clinical evaluation recommended  |
| 61% – 80%  | High Risk      | Significant cardiovascular risk markers present                  |
| 81% – 100% | Critical Risk  | Strong indication of severe cardiovascular risk factors          |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: Model file not found` | Ensure `retina_heart_model.h5` is in the project root directory |
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` inside your virtual environment |
| TensorFlow installation fails | Try `pip install tensorflow==2.15.0` specifically, or check [TensorFlow install guide](https://www.tensorflow.org/install) |
| App loads but model takes very long | First load is slow (~30–60s). The model is loaded once at server startup |
| Uploaded image rejected as "not retinal" | The validator checks for red/orange color dominance. Ensure you're uploading actual retinal fundus photographs |
| `Port 5000 already in use` | Edit `server.py` and change the port number in `app.run(port=5000)` |

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for medical advice.
