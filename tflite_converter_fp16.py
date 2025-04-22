"""
Converts a TensorFlow SavedModel (derived from RetinaFace ONNX) to TFLite Float16.

Handles potential dynamic input shapes in the SavedModel which might not be
correctly inferred by the TFLiteConverter by default.
"""

import os
import tensorflow as tf
import numpy as np

# --- Configuration ---
IMG_SIZE = 480 # Must match the input size used during ONNX/TF conversion

# Assumes script is run from the PARENT directory of 'tflite-retinaface2'
# Adjust paths if running from a different location.
PROJECT_ROOT = "tflite-retinaface2"
# This is the output directory from the onnx-tf conversion step
SAVED_MODEL_DIR_RELATIVE = os.path.join(PROJECT_ROOT, "weights", "retinaface_mobilefacenet025_tf")

TFLITE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "weights")
TFLITE_OUTPUT_FILENAME = f"{IMG_SIZE}-float16.tflite"
TFLITE_OUTPUT_PATH = os.path.join(TFLITE_OUTPUT_DIR, TFLITE_OUTPUT_FILENAME)
# --- End Configuration ---

def convert_tf_to_tflite_fp16():
    """Loads the SavedModel and converts it to TFLite Float16."""

    saved_model_dir_abs = os.path.abspath(SAVED_MODEL_DIR_RELATIVE)
    if not os.path.isdir(saved_model_dir_abs):
        print(f"Error: SavedModel directory not found at: {saved_model_dir_abs}")
        print("Please ensure the ONNX to TensorFlow conversion was successful.")
        return

    print(f"Attempting to load SavedModel from: {saved_model_dir_abs}")

    # Create the TFLite converter from the SavedModel.
    # It defaults to Float32 conversion initially.
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir_abs)

    # --- Configure for Float16 Conversion ---
    # Enable default optimizations, which includes FP16 quantization when target type is set.
    print("Configuring for Float16 conversion...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Specify Float16 as the target data type.
    converter.target_spec.supported_types = [tf.float16]
    # Specify supported operations. Including SELECT_TF_OPS can help if the model
    # contains TensorFlow operations not natively supported by TFLite.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Standard TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS    # Allow TF ops (Flex Delegate)
    ]

    # NOTE on Input Shape:
    # The SavedModel has dynamic input shape `(-1, 3, -1, -1)`.
    # TFLiteConverter might incorrectly create a static signature (e.g., `[1 3 1 1]`)
    # in the .tflite file.
    # Attempts to set the shape during conversion (e.g., via `input_shapes` argument
    # or `converter.inputs[0].set_shape()`) failed or caused errors in tested TF versions.
    # The workaround is to resize the input tensor *during inference* using the Interpreter API.
    # See the `test_tflite_fp16.py` script for an example.

    # Perform the conversion
    print("Starting TFLite conversion...")
    try:
        tflite_model = converter.convert()
        print("TFLite conversion successful.")
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
        return

    # Ensure output directory exists
    os.makedirs(TFLITE_OUTPUT_DIR, exist_ok=True)

    # Save the TFLite model
    print(f"Writing TFLite model to: {TFLITE_OUTPUT_PATH}")
    try:
        with open(TFLITE_OUTPUT_PATH, "wb") as f:
            f.write(tflite_model)
        print("TFLite model saved successfully.")
    except Exception as e:
        print(f"Error saving TFLite model: {e}")

if __name__ == "__main__":
    # Suppress excessive TensorFlow logging (0=all, 1=info, 2=warnings, 3=errors)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    convert_tf_to_tflite_fp16() 