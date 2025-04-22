"""
Tests a converted TFLite RetinaFace model (FP16 or FP32).

Includes the necessary workaround to resize the input tensor for models
converted from SavedModels with dynamic input shapes, where the TFLite
converter failed to preserve the correct input signature.
"""
import os
import time
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf # Use full TF to get Interpreter, can use tflite_runtime alternatively

def run_inference(model_path, image_path, num_threads=None):
    """Loads TFLite model, prepares input, runs inference, and prints results."""

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Load the TFLite model and allocate tensors.
    print(f"Loading model: {model_path}")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        print("Ensure TensorFlow or tflite_runtime is installed.")
        return

    # --- Input Tensor Resizing Workaround ---
    # This is CRITICAL because the TFLiteConverter often fails to correctly
    # set the input signature for models converted from TF SavedModels
    # that had dynamic input dimensions (like from onnx-tf).
    input_details_initial = interpreter.get_input_details()
    input_index = input_details_initial[0]['index']
    print(f"Initial input details: {input_details_initial}")

    # Define the expected NCHW shape (Batch, Channels, Height, Width)
    # Must match the dimensions used during conversion (e.g., 480x480)
    desired_shape = [1, 3, 480, 480] # Adjust if model uses different H/W
    print(f"Resizing input tensor {input_index} to shape: {desired_shape}")
    try:
        interpreter.resize_tensor_input(input_index, desired_shape)
    except Exception as e:
        print(f"Error resizing input tensor: {e}")
        return
    # --- Allocation must happen AFTER resizing ---
    print("Allocating tensors...")
    try:
        interpreter.allocate_tensors()
        print("Tensors allocated.")
    except Exception as e:
        print(f"Error allocating tensors: {e}")
        return

    # Get tensor details again AFTER resizing
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Input details AFTER resize: {input_details}")
    print(f"Output details AFTER resize: {output_details}")

    # Verify input type (should be float32 for FP16/FP32 models)
    is_floating_model = input_details[0]['dtype'] == np.float32
    print(f"Model input type: {input_details[0]['dtype']}")
    print(f"Is floating point model: {is_floating_model}")

    # --- Image Preprocessing ---
    # Get the required height and width from the (now resized) input tensor details.
    height = input_details[0]['shape'][2] # NCHW format
    width = input_details[0]['shape'][3]  # NCHW format
    print(f"Preparing image: Resizing to {width}x{height}")
    try:
        img = Image.open(image_path).convert('RGB').resize((width, height))
    except Exception as e:
        print(f"Error opening or resizing image: {e}")
        return

    # Convert image to numpy array, add batch dimension, and transpose to NCHW format.
    # Input format MUST match the model's expectation (NCHW for this model).
    input_data = np.expand_dims(img, axis=0)       # Shape: (1, H, W, C)
    input_data = input_data.transpose((0, 3, 1, 2)) # Shape: (1, C, H, W)

    # Ensure data type is Float32
    input_data = input_data.astype(np.float32)
    print(f"Actual input data shape: {input_data.shape}, dtype: {input_data.dtype}")

    # Apply normalization if it's a floating point model
    # These values might need adjustment depending on the original model training.
    input_mean = 127.5
    input_std = 127.5
    if is_floating_model:
        print(f"Applying normalization: mean={input_mean}, std={input_std}")
        input_data = (input_data - input_mean) / input_std
    else:
        print("Skipping normalization for non-floating model.")

    # --- Run Inference ---
    print("Running inference...")
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference (example: run once and time it)
    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()
    inference_time = stop_time - start_time

    # Get the output tensors
    # Adjust indices if the model output order differs
    # Output shapes depend on the model architecture and input size.
    # For RetinaFace with 480x480 input, expect N x 2 (conf), N x 10 (landms), N x 4 (loc)
    # where N is the number of anchor boxes (e.g., 9450).
    output_conf = interpreter.get_tensor(output_details[0]['index'])
    output_landms = interpreter.get_tensor(output_details[1]['index'])
    output_loc = interpreter.get_tensor(output_details[2]['index'])

    print("\n--- Inference Results ---")
    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"Output shapes:")
    print(f"  Confidence: {output_conf.shape}")
    print(f"  Landmarks:  {output_landms.shape}")
    print(f"  Locations:  {output_loc.shape}")
    # Further post-processing would be needed here to decode the outputs
    # into actual bounding boxes, landmarks, and confidence scores.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a TFLite RetinaFace model.")
    parser.add_argument(
        '-m',
        '--model_file',
        required=True,
        help='Path to the .tflite model file')
    parser.add_argument(
        '-i',
        '--image',
        required=True,
        help='Path to the input image file')
    parser.add_argument(
        '--num_threads', type=int, default=None, help='Number of threads for inference')

    args = parser.parse_args()

    # Suppress excessive TensorFlow logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    run_inference(args.model_file, args.image, args.num_threads) 