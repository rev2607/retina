"""
predictor.py
Handles image preprocessing, validation, and cardiovascular risk prediction.
Works with the multi-input EfficientNetB3 model (image + clinical data).
"""

import numpy as np
import cv2

# Model input dimensions
IMG_SIZE = 224


def is_retinal_image(image_rgb):
    """
    Validate if the provided image is likely a retinal fundus image based on HSV color profile.
    Retinal images typically have a dominant reddish-orange hue with moderate/high saturation
    and are not mostly grayscale or overly bright/dark.

    Args:
        image_rgb: np.ndarray, RGB image.

    Returns:
        bool: True if it passes validation, False otherwise.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Content mask (ignore very dark/black background)
    mask_content = v > 20
    content_pixels = np.sum(mask_content)
    
    # Check if the image is mostly black
    total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
    if content_pixels < (total_pixels * 0.1):
        return False
        
    sat_mean = np.mean(s[mask_content])
    
    # Check grayscale: if saturation is very low overall, it's not a color retina photo
    if sat_mean < 25: 
        return False
        
    # Check for reddish/orange dominance
    # H range in OpenCV (0-179): Red is 0-20 and 160-179, Orange is ~10-40
    mask_red_orange = ((h >= 0) & (h <= 45)) | ((h >= 150) & (h <= 179))
    mask_valid_color = mask_red_orange & mask_content
    color_ratio = np.sum(mask_valid_color) / content_pixels
    
    # If the majority of the content is red/orange, it's likely a retinal image
    if color_ratio > 0.35:
        return True
        
    # Fallback: Hough Circles check (look for the circular fundus mask)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=50, param2=30, minRadius=int(min(gray.shape)*0.2), maxRadius=max(gray.shape)
    )
    if circles is not None and len(circles) > 0:
        return True
        
    return False


def preprocess_image(uploaded_file):
    """
    Read and preprocess the uploaded image for model inference.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        tuple: (preprocessed_array, original_rgb_image)
    """
    # Read image bytes and decode
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(
            "Could not decode the uploaded file. "
            "Please upload a valid JPG, PNG, or JPEG image."
        )

    # Convert BGR → RGB for display and model input
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))

    # Convert to float32 and expand batch dimension
    # No /255 normalization — EfficientNetB3 handles this internally
    image_array = np.expand_dims(image_resized.astype(np.float32), axis=0)

    return image_array, image_rgb


def prepare_clinical_data(bp, cholesterol, bmi, diabetes):
    """
    Package clinical data into the format expected by the model.

    Args:
        bp (float): Systolic blood pressure (mmHg).
        cholesterol (float): Total cholesterol (mg/dL).
        bmi (float): Body Mass Index.
        diabetes (int): Diabetes status (0 or 1).

    Returns:
        np.ndarray: Shape (1, 4), dtype float32.
    """
    clinical = np.array([[bp, cholesterol, bmi, diabetes]], dtype=np.float32)
    return clinical


def predict(model, image_array, clinical_data):
    """
    Run model inference with image and clinical data.

    Args:
        model: Loaded Keras model.
        image_array: Preprocessed image array, shape (1, 224, 224, 3).
        clinical_data: Clinical features array, shape (1, 4).

    Returns:
        float: Predicted probability of cardiovascular risk (0-1).
    """
    # model() used instead of model.predict() to bypass macOS threading bugs
    prediction = model([image_array, clinical_data], training=False)
    probability = float(prediction[0][0])
    return probability


def interpret_risk(probability):
    """
    Interpret the predicted probability into a human-readable risk assessment.
    Updates the logic to provide fine-grained 20% intervals and dynamic UI colors.

    Args:
        probability (float): Model output probability (0-1).

    Returns:
        dict: Contains 'label', 'color', 'percent', and 'interpretation'.
    """
    pct = probability * 100

    # Color scheme constraints: Green <40%, Yellow 40-70%, Red >70%
    if pct < 40:
        ui_color = "#00C853" # Green
        emoji = "🟢"
    elif pct < 70:
        ui_color = "#FFD600" # Yellow
        emoji = "🟡"
    else:
        ui_color = "#FF4B4B" # Red
        emoji = "🔴"

    # Strict classification requirements:
    if pct <= 20:
        label = "Very Low Risk"
        interpretation = "Retinal vasculature appears exceptionally healthy with no noticeable risk factors."
    elif pct <= 40:
        label = "Low Risk"
        interpretation = "Healthy retina with minimal risk indicators. Routine monitoring advised."
    elif pct <= 60:
        label = "Moderate Risk"
        interpretation = "Borderline indicators detected. Lifestyle maintenance and further clinical evaluation recommended."
    elif pct <= 80:
        label = "High Risk"
        interpretation = "Significant cardiovascular risk markers present based on retinal vessel alterations."
    else:
        label = "Critical Risk"
        interpretation = "Strong indication of severe cardiovascular risk factors detected. Immediate consultation advised."

    return {
        "label": label,
        "color": ui_color,
        "percent": pct,
        "emoji": emoji,
        "interpretation": interpretation,
    }
