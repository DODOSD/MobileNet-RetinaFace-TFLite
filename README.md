# RetinaFace TFLite Conversion Documentation

This document outlines the steps taken to convert a PyTorch RetinaFace model (MobileNetV0.25 backbone) from the [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) repository (or similar forks like [tailtq/TFLite-RetinaFace](https://github.com/tailtq/TFLite-RetinaFace)) to the TensorFlow Lite (TFLite) Float16 format.

## Goal

The objective was to obtain a TFLite version of the RetinaFace face detection model suitable for deployment on edge devices or mobile platforms. We specifically targeted a Float16 model for a balance between size/performance and potential accuracy.

## Source Model

-   **Original Framework:** PyTorch
-   **Assumed Weights File:** `weights/mobilenet0.25_Final.pth` (relative to the `tflite-retinaface2` project root)
-   **Input Size:** 480x480 pixels
-   **Input Format (PyTorch/ONNX):** NCHW (Batch, Channels, Height, Width) - e.g., `[1, 3, 480, 480]`

## Conversion Steps

The conversion involved multiple stages: PyTorch -> ONNX -> TensorFlow SavedModel -> TFLite.

### 1. PyTorch -> ONNX

-   **Script:** `onnxconvert.py` (Included in this documentation folder)
-   **Tool:** `torch.onnx.export`
-   **Input:** `weights/mobilenet0.25_Final.pth`
-   **Output:** `weights/retinaface_mobilenet025.onnx`
-   **Command:**
    ```bash
    # Ensure PyTorch, ONNX, and other requirements are installed
    # (Ideally in a virtual environment)
    pip install torch torchvision torchaudio onnx onnxruntime opencv-python numpy
    # Adjust requirements based on the source repo's requirements.txt

    # Run the conversion script (assuming you are in the parent directory of tflite-retinaface2)
    python tflite-retinaface2/onnxconvert.py
    ```
-   **Notes:** This script defines the input/output names and specifies dynamic axes for batch size, height, and width.

### 2. ONNX -> TensorFlow SavedModel

-   **Tool:** `onnx-tf`
-   **Environment:** Docker container `tensorflow/tensorflow:devel-gpu` (Recommended by `tailtq/TFLite-RetinaFace` to ensure specific TF/CUDA/CuDNN versions and avoid conversion issues).
-   **Input:** `weights/retinaface_mobilenet025.onnx`
-   **Output:** `weights/retinaface_mobilefacenet025_tf/` (SavedModel directory)
-   **Command (Run from outside the container, e.g., PowerShell):**
    ```bash
    # Start an interactive container, mapping the project directory
    docker run -it --rm --gpus all -v /p/Python/emotion-try/tflite-retinaface2:/workspace/tflite-retinaface2 tensorflow/tensorflow:devel-gpu bash

    # --- Inside the container ---
    # Install required packages (TensorFlow should be pre-installed, but we needed to install it manually)
    pip install tensorflow onnx-tf

    # Navigate to the workspace
    cd /workspace/tflite-retinaface2

    # Run the conversion
    onnx-tf convert -i weights/retinaface_mobilenet025.onnx -o weights/retinaface_mobilefacenet025_tf

    # Exit the container
    exit
    # --- End of container commands ---
    ```
-   **Notes:** We encountered issues where the TensorFlow module wasn't found in the default container and had to install it manually. The volume mapping also required using an absolute path (`/p/Python/emotion-try/...`) in the `docker run` command to work correctly.

### 3. TensorFlow SavedModel -> TFLite (Float16)

-   **Script:** `tflite_converter_fp16.py` (Included in this documentation folder, adapted from the original `tflite_converter.py`)
-   **Tool:** `tf.lite.TFLiteConverter`
-   **Input:** `weights/retinaface_mobilefacenet025_tf/` (SavedModel directory)
-   **Output:** `weights/480-float16.tflite`
-   **Command:**
    ```bash
    # Ensure TensorFlow is installed in your local Python environment
    pip install tensorflow numpy

    # Run the conversion script
    python MobileNet-RetinaFace-TFLite/tflite_converter_fp16.py
    ```
-   **Notes:**
    -   The initial attempts using `TFLiteConverter` failed to correctly infer the input shape from the SavedModel (which had dynamic dimensions: `-1, 3, -1, -1`). The resulting TFLite models reported an incorrect fixed input shape like `[1 3 1 1]`.
    -   Attempts to quantize to INT8 also failed due to the shape inference issue.
    -   The final successful approach involved converting to Float16 without quantization settings in the script. The necessary input shape adjustment is handled during inference time (see Testing section).

## Testing the TFLite Model

-   **Script:** `test_tflite_fp16.py` (Included in this documentation folder, adapted from the original `test_tflite.py`)
-   **Input Model:** `weights/480-float16.tflite`
-   **Command:**
    ```bash
    # Ensure TensorFlow (or tflite-runtime) and Pillow are installed
    pip install tensorflow numpy Pillow

    # Run the test script
    python MobileNet-RetinaFace-TFLite/test_tflite_fp16.py -m tflite-retinaface2/weights/480-float16.tflite -i tflite-retinaface2/imgs/test-img2.jpeg
    ```
-   **CRITICAL WORKAROUND:** Because the TFLite model file (`.tflite`) ended up with an incorrect static input shape signature (`[1 3 1 1]`) despite the SavedModel having dynamic shapes, it is **essential** to resize the input tensor using the TFLite interpreter *after* loading the model but *before* allocating tensors and running inference. The included `test_tflite_fp16.py` script demonstrates this:
    ```python
    # [...] Load interpreter
    interpreter = tflite.Interpreter(model_path=args.model_file)

    # --- Resize input tensor BEFORE allocating ---
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    # Define desired NCHW shape (matching ONNX/SavedModel)
    desired_shape = [1, 3, 480, 480]
    interpreter.resize_tensor_input(input_index, desired_shape)

    # --- Allocate tensors AFTER resizing ---
    interpreter.allocate_tensors()
    # [...] Prepare input data in NCHW Float32 format
    # [...] Run inference
    ```

## Final Model

The final converted model is `480-float16.tflite`. Please place it in the `model/` subfolder within this documentation directory.

## License

The conversion scripts are based on code from the `tailtq/TFLite-RetinaFace` repository, which uses the MIT License. See the `LICENSE.MIT` file included in this folder. 

## Real-time Webcam Detection

A script `realtime_detect.py` is provided in this folder to run face detection on a live webcam feed using the converted TFLite model.

**Requirements:**

*   OpenCV (`opencv-python`)
*   NumPy
*   TensorFlow (`tensorflow`) or TensorFlow Lite Runtime (`tflite_runtime`)

```bash
pip install opencv-python numpy tensorflow Pillow
# Or, for only the runtime:
# pip install opencv-python numpy tflite_runtime Pillow
```

**Note on TensorFlow vs. TFLite Runtime:**
The script will attempt to use the lightweight `tflite_runtime` package first if it's installed. If `tflite_runtime` is not found or fails to import, it will fall back to the full `tensorflow` package. If you encounter issues with installing `tflite_runtime` (which can sometimes be tricky depending on the platform), you can force the script to use the full `tensorflow` package by adding the `--use_full_tf` flag. Using the full TensorFlow library might result in slightly slower startup times.

**Usage:**

```bash
# Make sure you have copied the 480-float16.tflite model into the MobileNet-RetinaFace-TFLite/model/ directory

# Run the script, providing the path to the model
python MobileNet-RetinaFace-TFLite/realtime_detect.py -m MobileNet-RetinaFace-TFLite/model/480-float16.tflite

# Optional arguments:
# --conf_thresh: Confidence threshold (default: 0.5)
# --nms_thresh: NMS threshold (default: 0.4)
# --num_threads: Number of threads for TFLite interpreter
# --use_full_tf: Force the use of the full TensorFlow library instead of tflite-runtime

# Example with different thresholds:
python MobileNet-RetinaFace-TFLite/realtime_detect.py -m MobileNet-RetinaFace-TFLite/model/480-float16.tflite --conf_thresh 0.6 --nms_thresh 0.3

# Example forcing full TensorFlow:
python MobileNet-RetinaFace-TFLite/realtime_detect.py -m MobileNet-RetinaFace-TFLite/model/480-float16.tflite --use_full_tf
```

Press 'q' in the display window to quit the application. 
