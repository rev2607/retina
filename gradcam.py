"""
gradcam.py
Grad-CAM implementation for the multi-input EfficientNetB3 model.
Automatically finds the last convolutional layer and generates
a heatmap overlay on the original retinal image.
"""

import numpy as np
import cv2
import tensorflow as tf


def find_last_conv_layer(model):
    """
    Automatically find the last Conv2D layer in the model.
    Iterates through all layers in reverse to find the deepest convolutional layer.

    Args:
        model: Loaded Keras model.

    Returns:
        str: Name of the last Conv2D layer.

    Raises:
        ValueError: If no Conv2D layer is found in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("No Conv2D layer found in the model for Grad-CAM.")


def generate_gradcam(model, image_array, clinical_data, target_class_index=0):
    """
    Generate a Grad-CAM heatmap for the given input.

    Creates a sub-model that outputs both the last conv layer activations
    and the final prediction, then uses GradientTape to compute gradients.

    Args:
        model: Loaded Keras model (multi-input).
        image_array: Preprocessed image, shape (1, 224, 224, 3).
        clinical_data: Clinical features, shape (1, 4).
        target_class_index: Index of the target output (0 for binary sigmoid).

    Returns:
        np.ndarray: Heatmap of shape (224, 224), values in [0, 1].
    """
    # Find the last convolutional layer
    last_conv_layer_name = find_last_conv_layer(model)
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Build a sub-model: same inputs → [last_conv_output, final_prediction]
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # Compute gradients using GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([image_array, clinical_data], training=False)
        loss = predictions[:, target_class_index]

    # Gradients of the prediction w.r.t. the last conv layer output
    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        # Fallback: return a uniform heatmap if gradients can't be computed
        return np.ones((224, 224), dtype=np.float32) * 0.5

    # Global average pooling of gradients → channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted sum of conv output feature maps
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU activation — keep only positive influences
    heatmap = tf.nn.relu(heatmap)

    # Normalize to [0, 1]
    heatmap_max = tf.reduce_max(heatmap)
    if heatmap_max > 0:
        heatmap = heatmap / heatmap_max

    heatmap = heatmap.numpy()

    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (224, 224))

    return heatmap


def overlay_heatmap(heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay the Grad-CAM heatmap on the original retinal image.

    Args:
        heatmap: Grad-CAM heatmap, shape (H, W), values in [0, 1].
        original_image: Original RGB image as np.ndarray.
        alpha: Blending factor for the overlay (0 = only original, 1 = only heatmap).
        colormap: OpenCV colormap to apply.

    Returns:
        np.ndarray: Superimposed RGB image (uint8).
    """
    # Resize original image to match heatmap
    original_resized = cv2.resize(original_image, (224, 224))

    # Convert heatmap to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert heatmap from BGR (OpenCV default) to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend: overlay = alpha * heatmap + (1-alpha) * original
    superimposed = cv2.addWeighted(heatmap_colored, alpha, original_resized, 1 - alpha, 0)

    return superimposed
