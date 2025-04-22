"""
Converts a PyTorch RetinaFace model to ONNX format.

Based on conversion steps from https://github.com/tailtq/TFLite-RetinaFace
Assumes the source project structure and necessary weights files are present.
"""

import torch
import os

# Attempt to import necessary modules from the source project structure
try:
    from data import cfg_mnet # Configuration for MobileNetV0.25 backbone
    from models.retinaface import RetinaFace # Model definition
    from detect import load_model # Utility to load PyTorch weights
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure this script is run from a directory containing the ")
    print("'data', 'models', and 'detect' modules from the source project, ")
    print("or adjust the import paths accordingly.")
    exit(1)

# --- Configuration ---
# Assumes script is run from the PARENT directory of 'tflite-retinaface2'
# Adjust paths if running from a different location.
PROJECT_ROOT = "tflite-retinaface2"
PYTORCH_MODEL_PATH = os.path.join(PROJECT_ROOT, "weights/mobilenet0.25_Final.pth")
ONNX_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "weights")
ONNX_OUTPUT_FILENAME = "retinaface_mobilenet025.onnx"
ONNX_OUTPUT_PATH = os.path.join(ONNX_OUTPUT_DIR, ONNX_OUTPUT_FILENAME)

INPUT_HEIGHT = 480
INPUT_WIDTH = 480
# --- End Configuration ---

def convert_pytorch_to_onnx():
    """Loads the PyTorch model and exports it to ONNX."""
    if not os.path.exists(PYTORCH_MODEL_PATH):
        print(f"Error: PyTorch weights file not found at: {PYTORCH_MODEL_PATH}")
        print("Please ensure the weights file exists.")
        return

    print(f"Loading PyTorch model from: {PYTORCH_MODEL_PATH}")
    # Instantiate the model structure (using mobilenet config)
    net = RetinaFace(cfg=cfg_mnet, phase='test')

    # Load the trained weights (forcing load to CPU for broader compatibility)
    # Change load_to_cpu=False if using a GPU during conversion.
    net = load_model(net, PYTORCH_MODEL_PATH, load_to_cpu=True)
    net.eval() # Set the model to evaluation mode (important for export)
    print("PyTorch model loaded successfully.")

    # Create a dummy input tensor with the expected shape (NCHW format)
    # Batch size = 1, Channels = 3 (RGB), Height, Width
    dummy_input = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH, requires_grad=False)
    print(f"Creating dummy input with shape: {dummy_input.shape}")

    # Ensure output directory exists
    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)

    print(f"Exporting model to ONNX format at: {ONNX_OUTPUT_PATH}")
    try:
        torch.onnx.export(
            net,                                # The model to export
            dummy_input,                        # Example input tensor
            ONNX_OUTPUT_PATH,                   # Where to save the model
            export_params=True,                 # Store trained weights within the model file
            opset_version=11,                   # ONNX version (11 is widely compatible)
            do_constant_folding=True,           # Apply optimizations during export
            input_names=['input'],             # Name for the input node
            output_names=['loc', 'conf', 'landms'], # Names for the output nodes (adjust if model differs)
            dynamic_axes={                      # Specify axes with variable dimensions
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'loc': {0: 'batch_size'},
                'conf': {0: 'batch_size'},
                'landms': {0: 'batch_size'}
            }
        )
        print("ONNX export complete.")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

if __name__ == "__main__":
    convert_pytorch_to_onnx() 