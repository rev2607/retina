# 👁️ RetinaRisk — Cardiovascular Risk Prediction from Retinal Images

An AI-powered web application that predicts cardiovascular risk from retinal fundus images using a pre-trained **EfficientNetB3** deep learning model with **Grad-CAM** explainability.

---

## 🎯 Features

- **Retinal Image Analysis** — Upload a fundus photograph and get an instant cardiovascular risk score
- **Multi-Modal Prediction** — Combines retinal image features with clinical data (BP, cholesterol, BMI, diabetes)
- **Grad-CAM Visualization** — See which regions of the retina the model focuses on
- **Downloadable Heatmaps** — Export the Grad-CAM overlay as a PNG
- **Premium UI** — Clean, dark-themed Streamlit interface with real-time results

---

## 🛠️ Setup

### Prerequisites
- Python 3.9+
- The pre-trained model file `retina_heart_model.h5` (must be in the project root)

### Installation

```bash
# Clone or navigate to the project directory
cd /path/to/retina

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
retina/
├── app.py               # Main Streamlit web application
├── model_loader.py      # Model loading with caching
├── predictor.py         # Image preprocessing & prediction logic
├── gradcam.py           # Grad-CAM heatmap generation
├── requirements.txt     # Python dependencies
├── retina_heart_model.h5  # Pre-trained model (~96 MB)
└── README.md            # This file
```

---

## 📊 Model Architecture

| Component         | Detail                                      |
|--------------------|---------------------------------------------|
| **Backbone**       | EfficientNetB3 (ImageNet pre-trained)       |
| **Image Input**    | 224 × 224 × 3 RGB                           |
| **Clinical Input** | 4 features (BP, Cholesterol, BMI, Diabetes) |
| **Output**         | Single sigmoid neuron (0–1 probability)     |
| **Threshold**      | 0.5 (≥ 0.5 = High Risk)                    |

---

## 🖥️ Example Usage

1. Open the app with `streamlit run app.py`
2. Adjust clinical parameters in the sidebar (or leave defaults)
3. Upload a retinal fundus image (JPG/PNG)
4. View the results:
   - **Risk Score** — Probability from 0 to 1
   - **Risk Label** — "Low Risk" (green) or "High Cardiovascular Risk" (red)
   - **Grad-CAM** — Heatmap overlay highlighting important retinal regions
   - **Interpretation** — Plain-text explanation of the risk level

### Output Interpretation

| Score Range | Interpretation                                                |
|-------------|---------------------------------------------------------------|
| 0.00 – 0.29 | Retinal vasculature appears healthy                           |
| 0.30 – 0.49 | Mild indicators present — routine monitoring advised          |
| 0.50 – 0.59 | Borderline — further clinical evaluation recommended          |
| 0.60 – 0.79 | Moderate-to-high cardiovascular risk markers                  |
| 0.80 – 1.00 | Strong indication of cardiovascular risk factors              |

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for medical advice.
