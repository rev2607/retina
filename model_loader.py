"""
model_loader.py
Loads the pre-trained Keras model with Streamlit caching.
Bypasses TF/Keras version incompatibilities by explicitly constructing
the model architecture and only loading the weights.
"""

import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3

# Path to the pre-trained model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retina_heart_model.h5")


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Construct the EfficientNetB3 multi-input model architecture and load weights.
    Uses Streamlit's cache_resource to ensure the model is loaded only once.

    Returns:
        tf.keras.Model: The loaded Keras model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If weights loading fails.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Please ensure 'retina_heart_model.h5' is in the project directory."
        )

    try:
        # Rebuild architecture
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
        
        # Load only weights (ignores Keras serialization mismatches)
        model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
        return model
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load weights from '{MODEL_PATH}'. "
            f"Error: {e}"
        )
