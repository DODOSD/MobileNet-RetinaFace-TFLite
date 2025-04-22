"""
Real-time face detection using a TFLite RetinaFace model and webcam.

Based on processing steps from:
- https://github.com/biubug6/Pytorch_Retinaface
- https://github.com/tailtq/TFLite-RetinaFace

Includes the necessary workaround to resize the input tensor for models
converted from SavedModels with dynamic input shapes, where the TFLite
converter failed to preserve the correct input signature.
"""

import cv2
import time
import argparse
import numpy as np
from PIL import Image
# Use tflite_runtime for smaller dependency and potentially faster startup
# import tensorflow as tf
# import tflite_runtime.interpreter as tflite # REMOVE this top-level import
from itertools import product
from math import ceil
import os
# --- Add threading and queue imports ---
import threading
from queue import Queue, Empty
# --- End imports ---

# --- Post-processing Utilities (Adapted from original PyTorch code to NumPy) ---

# Configuration for MobileNetV0.25 backbone (ensure this matches the trained model)
cfg_mnet = {
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2], # Variance for box decoding
    'clip': False, # Clip box coordinates to [0, 1]?
    'image_size': 480 # Input image size
}

class PriorBoxNumpy(object):
    """Generates prior anchor boxes similar to the original PriorBox layer."""
    def __init__(self, cfg, image_size=None):
        super(PriorBoxNumpy, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output = np.clip(output, 0, 1)
        return output

def decode_boxes_numpy(loc, priors, variances):
    """Decode locations from predictions using priors to undo encoding."""
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2 # Convert center to xmin, ymin
    boxes[:, 2:] += boxes[:, :2]     # Convert width, height to xmax, ymax
    return boxes

def decode_landm_numpy(pre, priors, variances):
    """Decode landmarks from predictions using priors."""
    landms = np.concatenate((
        priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ), axis=1)
    return landms

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    if dets.shape[0] == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# --- Webcam Stream Class ---
class WebcamStream:
    """Reads frames from webcam in a separate thread with optimizations."""
    def __init__(self, src=0):
        self.src = src
        # Determine backend based on OS
        self.backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        print(f"Using VideoCapture backend: {'DSHOW' if self.backend == cv2.CAP_DSHOW else 'Default'}")

        self.stream = None # Initialize later
        self.queue = Queue(maxsize=1)
        self.stopped = False
        # Start initialization in a separate thread
        self.init_thread = threading.Thread(target=self._initialize_camera, daemon=True)
        self.capture_thread = None # Initialize after camera is ready
        self.camera_ready = threading.Event() # To signal when camera is open

    def _initialize_camera(self):
        """Opens camera and sets properties in a background thread."""
        print("Camera initialization thread started...")
        try:
            self.stream = cv2.VideoCapture(self.src, self.backend)
            if not self.stream.isOpened():
                 # Fallback if DSHOW fails
                 print("Warning: Initial backend failed, trying default...")
                 self.stream = cv2.VideoCapture(self.src)
                 if not self.stream.isOpened():
                     raise IOError(f"Cannot open webcam {self.src} with any backend.")

            # Apply settings AFTER opening
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Use a common default initially
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # self.stream.set(cv2.CAP_PROP_FPS, 30) # Setting FPS often has little effect
            self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0) # Disable autofocus
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce buffer

            # Try a single grab to confirm it works
            if not self.stream.grab():
                raise IOError(f"Cannot grab frame from webcam {self.src}.")

            print("Camera successfully initialized and grabbed first frame.")
            self.camera_ready.set() # Signal that camera is ready

            # Start the frame reading thread now
            self.capture_thread = threading.Thread(target=self._update, daemon=True)
            self.capture_thread.start()

        except Exception as e:
            print(f"!!! Camera Initialization Error: {e} !!!")
            self.stopped = True # Ensure update loop doesn't run
            if self.stream:
                self.stream.release()
            # We can optionally set the event even on error so start() doesn't block forever,
            # but the error state should be checked.
            self.camera_ready.set() # Signal completion even if error occurred

    def start(self):
        print("Requesting camera start...")
        self.stopped = False
        self.init_thread.start()
        # Optional: Wait for init thread to signal ready, or let main loop handle it
        # self.camera_ready.wait(timeout=5) # Wait max 5s for camera
        # if not self.camera_ready.is_set():
        #     print("Warning: Camera initialization timed out or failed.")
        return self

    def _update(self):
        """Internal method run by the thread to continuously read frames."""
        print("Frame reading thread started...")
        while not self.stopped:
            if not self.stream or not self.stream.isOpened():
                # Camera closed unexpectedly or failed init
                time.sleep(0.1)
                continue
            try:
                ret, frame = self.stream.read()
                if not ret:
                    print("Warning: Failed to read frame from stream.")
                    time.sleep(0.05) # Wait briefly before retrying
                    continue

                if not self.queue.empty():
                    try: self.queue.get_nowait()
                    except Queue.empty: pass
                self.queue.put(frame)
            except Exception as e:
                 print(f"Error in frame reading thread: {e}")
                 time.sleep(0.1)
        print("Frame reading thread finished.")

    def read(self):
        """Return the latest frame from the queue."""
        # If camera init failed, return None immediately
        if self.stopped and not self.camera_ready.is_set():
            return None
        # Wait briefly if camera not ready yet
        if not self.camera_ready.is_set():
             ready = self.camera_ready.wait(timeout=0.1) # Short wait
             if not ready:
                  # print("Waiting for camera...")
                  return None # Still initializing
        # Camera should be ready or init failed
        try:
            return self.queue.get(block=False) # Use block=False or small timeout
        except Empty:
            return None

    def stop(self):
        """Signal the threads to stop and release resources."""
        print("Stopping webcam stream...")
        self.stopped = True
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
             self.capture_thread.join(timeout=1.0)
        if self.init_thread and self.init_thread.is_alive():
             self.init_thread.join(timeout=1.0)
        if self.stream:
            self.stream.release()
        print("Webcam stream stopped.")
# --- End Webcam Stream Class ---

# --- Main Detection Logic ---

def run_realtime_detection(model_path, conf_threshold=0.5, nms_threshold=0.4, num_threads=None):
    """Loads model and runs detection loop on webcam feed using threaded reader."""

    # --- Model Loading & Setup (same as before) ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model: {model_path}")
    try:
        # Use tflite.Interpreter instead of tf.lite.Interpreter
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        # Make sure tflite_runtime is installed if tensorflow isn't
        print("Ensure tensorflow or tflite_runtime is installed.")
        return

    # --- Input Tensor Resizing Workaround ---
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']

    # --- Correct Output Indices (IMPORTANT!) ---
    CONF_INDEX = 583
    LANDMS_INDEX = 597
    LOC_INDEX = 568
    print(f"Using output indices - Conf: {CONF_INDEX}, Landms: {LANDMS_INDEX}, Loc: {LOC_INDEX}")
    output_indices_from_model = {detail['index'] for detail in output_details}
    if not {CONF_INDEX, LANDMS_INDEX, LOC_INDEX}.issubset(output_indices_from_model):
        print("Error: One or more specified output indices not found in the model!")
        print("Actual output details:", output_details)
        return
    # --- End Correct Output Indices ---

    print(f"Initial input details: {input_details}")
    desired_shape = [1, 3, 480, 480] # NCHW format
    print(f"Resizing input tensor {input_index} to shape: {desired_shape}")
    try:
        interpreter.resize_tensor_input(input_index, desired_shape)
    except Exception as e:
        print(f"Error resizing input tensor: {e}")
        return
    print("Allocating tensors...")
    try:
        interpreter.allocate_tensors()
        print("Tensors allocated.")
    except Exception as e:
        print(f"Error allocating tensors: {e}")
        return

    input_details = interpreter.get_input_details() # Get details again after resize
    print(f"Input details AFTER resize: {input_details}")
    input_height = input_details[0]['shape'][2]
    input_width = input_details[0]['shape'][3]

    # Check if model expects float input
    is_floating_model = input_details[0]['dtype'] == np.float32
    print(f"Model expects float32 input: {is_floating_model}")

    # --- Prepare Prior Boxes (Generate ONCE before the loop) ---
    print("Generating prior boxes...")
    try:
        priorbox = PriorBoxNumpy(cfg=cfg_mnet, image_size=(input_height, input_width))
        priors = priorbox.forward()
        print(f"Generated {priors.shape[0]} prior boxes.")
    except Exception as e:
        print(f"Error generating prior boxes: {e}")
        return
    # --- End Prepare Prior Boxes ---

    # --- Webcam Setup using Threaded Stream ---
    print("Initializing webcam stream...")
    try:
        vs = WebcamStream(src=0).start()
    except IOError as e:
        print(e)
        return
    # Allow camera sensor to warm up - Removed sleep time entirely
    # time.sleep(0.2) # Removed
    # print("Webcam stream started.") # Moved to _initialize_camera
    # --- End Webcam Setup ---

    frame_count = 0
    fps = 0
    start_time = time.time()
    process_start_time = start_time # For overall processing time

    while True:
        # --- Read frame from threaded stream ---
        frame = vs.read()
        if frame is None:
            # If queue is empty, maybe wait briefly or skip frame
            # print("Skipping frame, queue empty.")
            # time.sleep(0.01)
            continue # Skip processing if no frame available
        # --- End Read frame ---

        # --- Preprocessing & Inference (same as before) ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = frame.shape
        img_resized = cv2.resize(frame_rgb, (input_width, input_height))

        # Prepare input tensor (NCHW format, Float32)
        input_data = np.expand_dims(img_resized, axis=0)
        input_data = input_data.transpose((0, 3, 1, 2)) # HWC to CHW -> NCHW
        input_data = input_data.astype(np.float32)

        # --- Apply Mean Subtraction (Matching common RetinaFace practice) ---
        # input_mean = 127.5
        # input_std = 127.5
        # if is_floating_model:
        #     input_data = (input_data - input_mean) / input_std
        # Instead of normalization, subtract channel means (BGR order means for NCHW input)
        # Ensure means are broadcastable to (1, 3, 1, 1) for NCHW format
        channel_means = np.array([[[[104]], [[117]], [[123]]]], dtype=np.float32) # BGR means reshaped for NCHW
        if is_floating_model: # Only apply if model expects float
            # print("Applying mean subtraction (104, 117, 123)...") # Commented out
            input_data -= channel_means
        # --- End Mean Subtraction ---

        # --- Inference ---
        interpreter.set_tensor(input_index, input_data)
        inference_start = time.time()
        interpreter.invoke()
        inference_time_ms = (time.time() - inference_start) * 1000

        # --- Post-processing & Drawing (same as before) ---
        loc = np.squeeze(interpreter.get_tensor(LOC_INDEX))
        conf = np.squeeze(interpreter.get_tensor(CONF_INDEX))
        landms = np.squeeze(interpreter.get_tensor(LANDMS_INDEX))
        # Decode boxes and landmarks
        boxes = decode_boxes_numpy(loc, priors, cfg_mnet['variance'])
        boxes = boxes * np.array([img_width, img_height, img_width, img_height])
        landms = decode_landm_numpy(landms, priors, cfg_mnet['variance'])
        landms = landms * np.array([img_width, img_height] * 5)
        scores = conf[:, 1]

        # Ignore low confidence detections
        valid_idx = np.where(scores > conf_threshold)[0]
        # print(f"Detections above conf threshold ({conf_threshold}): {len(valid_idx)}") # Commented out
        boxes = boxes[valid_idx]
        landms = landms[valid_idx]
        scores = scores[valid_idx]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # Drawing
        for b in dets:
            # Cast coordinates to int, keep score as float
            x1, y1, x2, y2 = map(int, b[0:4])
            score = b[4]
            # Draw bounding box (blue)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Put confidence score (use float score)
            text = f"{score:.2f}"
            # Change text color to yellow for better visibility
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        # Draw landmarks (optional)

        # --- FPS Calculation & Display ---
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= 1.0: # Update FPS approx every second
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time

        # Display FPS and Inference Time
        fps_text = f"FPS: {fps:.1f}"
        inf_time_text = f"Inference: {inference_time_ms:.1f} ms"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, inf_time_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Real-time RetinaFace Detection', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check total processing time if needed (optional)
        # total_elapsed = time.time() - process_start_time

    # --- Cleanup ---
    print("Stopping webcam stream and closing windows...")
    vs.stop() # Stop the webcam reading thread
    cv2.destroyAllWindows()
    print("Cleanup complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time face detection with TFLite RetinaFace.")
    parser.add_argument(
        '-m',
        '--model_path',
        required=True,
        help='Path to the .tflite model file')
    parser.add_argument(
        '--conf_thresh', type=float, default=0.5,
        help='Confidence threshold for filtering detections')
    parser.add_argument(
        '--nms_thresh', type=float, default=0.4,
        help='Non-Maximum Suppression (NMS) threshold')
    parser.add_argument(
        '--num_threads', type=int, default=None,
        help='Number of threads for TFLite interpreter')
    # Add argument to potentially use full TF if needed
    parser.add_argument(
        '--use_full_tf', action='store_true',
        help='Use full TensorFlow library instead of tflite_runtime (requires tensorflow package)')

    args = parser.parse_args()

    # Dynamically import based on argument
    if args.use_full_tf:
        print("Using full TensorFlow library.")
        import tensorflow as tf
        tflite = tf.lite
    else:
        try:
            print("Using tflite_runtime library.")
            import tflite_runtime.interpreter as tflite
        except ImportError:
            print("Error: tflite_runtime not found. Install it (`pip install tflite-runtime`) or use full TensorFlow (`pip install tensorflow --upgrade` and run with --use_full_tf flag).")
            exit()

    run_realtime_detection(args.model_path, args.conf_thresh, args.nms_thresh, args.num_threads)