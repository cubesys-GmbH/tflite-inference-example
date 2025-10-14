import os
import cv2
import time
import numpy as np
import argparse

from PIL import Image  
from tflite_runtime.interpreter import Interpreter


def load_interpreter(model_path: str, use_delegate: bool) -> Interpreter:
    delegates = []
    
    if use_delegate:
        # Try VX delegate
        try:
            from tflite_runtime.interpreter import load_delegate
            vx_delegate = load_delegate('/usr/lib/libvx_delegate.so')
            delegates.append(vx_delegate)
            print("VX delegate loaded (NPU acceleration enabled)")
        except Exception as e:
            print(e)
            print("Running on CPU fallback")
    else:
        print("Running inference on CPU (delegate disabled)")

    from multiprocessing import cpu_count
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=delegates,
        num_threads = cpu_count(),
    )
    return interpreter


def resize_image(cv_image: np.ndarray, height: int, width: int) -> np.ndarray:
    color_converted = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted)
    image_resized = pil_image.resize((width, height))
    return image_resized


def draw_bounding_box(cv_image: np.ndarray, detections: list, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    image = cv_image.copy()
    frame_height, frame_width, _ = cv_image.shape
    for detection in detections:
        y1, x1, y2, x2 = detection[2]

        x1 = int(x1 * frame_width)
        x2 = int(x2 * frame_width)
        y1 = int(y1 * frame_height)
        y2 = int(y2 * frame_height)

        top = max(0, np.floor(y1 + 0.5))
        left = max(0, np.floor(x1 + 0.5))
        bottom = min(frame_height, np.floor(y2 + 0.5))
        right = min(frame_width, np.floor(x2 + 0.5))

        cv2.rectangle(cv_image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    return cv_image


# --- Command-line arguments ---
parser = argparse.ArgumentParser(description="TFLite inference on cube:evk")
parser.add_argument("--input", type=str, default="input/example.jpg", help="Path to input image")
parser.add_argument("--output", type=str, default="output/result.jpg", help="Path to save output image")
parser.add_argument("--no-delegate", action="store_true", help="Run inference without VX delegate (CPU only)")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

MODEL_PATH = "models/ssd_mobilenet_v1_1/ssd_mobilenet_v1_1.tflite"
INPUT_IMAGE = args.input
OUTPUT_PATH = args.output

# --- Load interpreter ---
interpreter = load_interpreter(MODEL_PATH, use_delegate=not args.no_delegate)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

warmup_start = time.time()
interpreter.invoke()
warmup_end = time.time()
print(f"Interpreter warmup time: {warmup_end-warmup_start:.2f} sec")

# --- Load and preprocess image ---
cv_image = cv2.imread(INPUT_IMAGE)
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

image_resized = resize_image(cv_image=cv_image, height=input_height, width=input_width)
image_batch = np.expand_dims(image_resized, axis=0)

# Normalize if model expects float
if input_details[0]['dtype'] == np.float32:
    input_data = image_batch / 255.0

inference_start = time.time() 
interpreter.set_tensor(input_details[0]['index'], image_batch)
interpreter.invoke()
inference_end = time.time()

# --- Post-processing (SSD-style outputs) ---
boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
classes = np.squeeze(interpreter.get_tensor(output_details[1]['index'])).astype(int)
confidence = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

# --- Draw detections ---
detections = []
for idx, class_id in enumerate(classes):
    if confidence[idx] > 0.6:
        detections.append((class_id, confidence[idx], boxes[idx]))
print(detections)
frame = draw_bounding_box(cv_image, detections)

# --- Save output ---
cv2.imwrite(OUTPUT_PATH, frame)
print(f"Inference complete in {inference_end-inference_start:.3f} sec. Output saved at {OUTPUT_PATH}")

