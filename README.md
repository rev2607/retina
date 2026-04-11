# 👁️ RetinaRisk — Cardiovascular Risk Prediction from Retinal Images

An AI-powered web application that predicts cardiovascular risk from retinal fundus images using a pre-trained **EfficientNetB3** deep learning model with **Grad-CAM** explainability.

---

## 🎯 Features

- **Retinal Image Analysis** — Upload a fundus photograph and get an instant cardiovascular risk score
- **Multi-Modal Prediction** — Combines retinal image features with clinical data (BP, cholesterol, BMI, diabetes)
- **Grad-CAM Visualization** — See which regions of the retina the model focuses on
- **Downloadable Heatmaps** — Export the Grad-CAM overlay as a PNG
- **Batch Upload** — Analyze multiple retinal images at once
- **Image Validation** — Automatically rejects non-retinal images
- **Premium UI** — Clean, dark-themed Streamlit interface with real-time results

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
| `streamlit` | ≥ 1.30.0 | Web application framework |
| `tensorflow` | ≥ 2.15.0 | Deep learning model & inference |
| `opencv-python-headless` | ≥ 4.8.0 | Image processing & Grad-CAM overlay |
| `numpy` | ≥ 1.24.0 | Numerical computations |
| `Pillow` | ≥ 10.0.0 | Image format conversion for downloads |

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
├── app.py
├── retina_heart_model.h5   ← This file must be present
├── ...
```

### Step 5: Launch the Application

```bash
streamlit run app.py
```

The app will open automatically in your default browser at **`http://localhost:8501`**.

> **First launch** may take 30–60 seconds while the model loads into memory. Subsequent analyses are near-instant thanks to caching.

---

## 📁 Project Structure

```
retina/
├── app.py                   # Main Streamlit web application
├── model_loader.py          # Model loading with caching
├── predictor.py             # Image preprocessing, validation & prediction logic
├── gradcam.py               # Grad-CAM heatmap generation
├── requirements.txt         # Python dependencies
├── retina_heart_model.h5    # Pre-trained model weights (~96 MB)
├── retina_(1)_(1).ipynb     # Jupyter notebook (model training & experimentation)
├── kaggle (1).json          # Kaggle API credentials for dataset access
├── documentation.md         # List of output screenshots
├── explanation.md           # Detailed project explanation
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

1. Open the app with `streamlit run app.py`
2. Adjust clinical parameters in the sidebar (or leave defaults):
   - Systolic Blood Pressure (80–200 mmHg)
   - Total Cholesterol (100–400 mg/dL)
   - BMI (15.0–45.0)
   - Diabetes Status (Yes / No)
3. Upload one or more retinal fundus images (JPG / PNG)
4. View the results:
   - **Risk Score** — Probability from 0% to 100%
   - **Risk Label** — Color-coded risk tier (Very Low → Critical)
   - **Grad-CAM** — Heatmap overlay highlighting important retinal regions
   - **Interpretation** — Plain-text explanation of the risk level
5. Download the Grad-CAM heatmap overlay if needed

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
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements.txt` inside your virtual environment |
| TensorFlow installation fails | Try `pip install tensorflow==2.15.0` specifically, or check [TensorFlow install guide](https://www.tensorflow.org/install) |
| App loads but model takes very long | First load is slow (~30–60s). Subsequent runs use cached model. Ensure at least 4 GB RAM is free |
| Uploaded image rejected as "not retinal" | The validator checks for red/orange color dominance. Ensure you're uploading actual retinal fundus photographs |
| `Port 8501 already in use` | Run `streamlit run app.py --server.port 8502` to use a different port |

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for medical advice.
