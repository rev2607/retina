"""
server.py
Flask backend for RetinaRisk web application.
Serves static frontend and provides /api/analyze endpoint for batch image analysis.
"""

import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

from model_loader import load_model
from predictor import preprocess_image, prepare_clinical_data, predict, interpret_risk, is_retinal_image
from gradcam import generate_gradcam, overlay_heatmap

# ─── App Setup ───────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="")

# Load model once at startup
print("[*] Loading AI model...")
# We need a custom load since we're not using Streamlit's cache
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retina_heart_model.h5")

def _load_model():
    """Load model without Streamlit dependency."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")
    img_input = layers.Input(shape=(224, 224, 3), name="img_input")
    base_model = EfficientNetB3(weights=None, include_top=False, input_tensor=img_input)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    clinical_input = layers.Input(shape=(4,), name="clinical_input")
    c = layers.Dense(64, activation="relu")(clinical_input)
    c = layers.BatchNormalization()(c)
    c = layers.Dense(32, activation="relu")(c)
    combined = layers.concatenate([x, c])
    z = layers.Dense(128, activation="relu")(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(32, activation="relu")(z)
    output = layers.Dense(1, activation="sigmoid")(z)
    model = models.Model(inputs=[img_input, clinical_input], outputs=output)
    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
    return model

model = _load_model()
print("[OK] Model loaded successfully!")


# ─── Helper ──────────────────────────────────────────────────────────────────────
IMG_SIZE = 224

def numpy_to_base64(img_array):
    """Convert a numpy RGB image to a base64-encoded PNG string."""
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def process_single_image(file_storage, clinical_data):
    """Process one uploaded image: validate, predict, generate Grad-CAM."""
    result = {"name": file_storage.filename, "error": None}

    # Read and decode
    file_bytes = file_storage.read()
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        result["error"] = "Could not decode image. Upload a valid JPG/PNG file."
        return result

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Validate
    if not is_retinal_image(image_rgb):
        result["error"] = "Not recognized as a retinal fundus image."
        result["original"] = numpy_to_base64(cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE)))
        return result

    # Preprocess
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_array = np.expand_dims(image_resized.astype(np.float32), axis=0)

    # Predict
    probability = predict(model, image_array, clinical_data)
    risk = interpret_risk(probability)

    # Grad-CAM
    gradcam_b64 = None
    try:
        heatmap = generate_gradcam(model, image_array, clinical_data)
        gradcam_img = overlay_heatmap(heatmap, image_rgb)
        gradcam_b64 = numpy_to_base64(gradcam_img)
    except Exception:
        pass

    result.update({
        "original": numpy_to_base64(image_resized),
        "probability": probability,
        "risk": risk,
        "gradcam": gradcam_b64,
    })
    return result


# ─── Routes ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Accepts multipart form data:
      - images: multiple image files
      - bp, cholesterol, bmi, diabetes: clinical parameters
    Returns JSON with results for each image.
    """
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No images uploaded."}), 400

    bp = float(request.form.get("bp", 120))
    cholesterol = float(request.form.get("cholesterol", 200))
    bmi = float(request.form.get("bmi", 25.0))
    diabetes = int(request.form.get("diabetes", 0))

    clinical_data = prepare_clinical_data(bp, cholesterol, bmi, diabetes)

    results = []
    for f in files:
        if f.filename == "":
            continue
        r = process_single_image(f, clinical_data)
        results.append(r)

    return jsonify({"results": results})


# ─── Run ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, port=5000)
