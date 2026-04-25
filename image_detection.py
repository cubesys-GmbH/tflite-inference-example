# Copyright (c) 2025 cubesys GmbH
# Licensed under the MIT License. See LICENSE for details.

import argparse
import os
import sys
import time
from multiprocessing import cpu_count

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

VX_DELEGATE_PATH = "/usr/lib/libvx_delegate.so"
DEFAULT_MODEL = "models/ssd_mobilenet_v1_1/ssd_mobilenet_v1_1.tflite"
DEFAULT_LABELS = "models/ssd_mobilenet_v1_1/labels.txt"
CONFIDENCE_THRESHOLD = 0.6


def load_interpreter(model_path: str, use_delegate: bool) -> Interpreter:
    delegates = []
    if use_delegate:
        try:
            delegates.append(load_delegate(VX_DELEGATE_PATH))
            print("VX delegate loaded (NPU acceleration enabled)")
        except Exception as e:
            print(e)
            print("Running on CPU fallback")
    else:
        print("Running inference on CPU (delegate disabled)")

    return Interpreter(
        model_path=model_path,
        experimental_delegates=delegates,
        num_threads=cpu_count(),
    )


def load_labels(path: str) -> dict:
    labels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1]
    return labels


def preprocess(cv_image: np.ndarray, height: int, width: int, dtype) -> np.ndarray:
    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height))
    batch = np.expand_dims(resized, axis=0)
    if dtype == np.float32:
        batch = (batch / 255.0).astype(np.float32)
    return batch


def parse_ssd_outputs(interpreter: Interpreter, output_details: list):
    # Standard TFLite SSD postprocess names; fall back to positional ordering.
    by_name = {od['name']: od for od in output_details}
    base = "TFLite_Detection_PostProcess"
    if base in by_name:
        boxes_d = by_name[base]
        classes_d = by_name[f"{base}:1"]
        scores_d = by_name[f"{base}:2"]
    else:
        boxes_d, classes_d, scores_d = output_details[:3]

    boxes = np.squeeze(interpreter.get_tensor(boxes_d['index']))
    classes = np.squeeze(interpreter.get_tensor(classes_d['index'])).astype(int)
    scores = np.squeeze(interpreter.get_tensor(scores_d['index']))
    return boxes, classes, scores


def draw_bounding_box(cv_image: np.ndarray, detections: list, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    image = cv_image.copy()
    h, w = image.shape[:2]
    for _, _, box in detections:
        y1, x1, y2, x2 = box
        x1 = max(0, min(w, int(x1 * w)))
        x2 = max(0, min(w, int(x2 * w)))
        y1 = max(0, min(h, int(y1 * h)))
        y2 = max(0, min(h, int(y2 * h)))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="TFLite inference on cube:evk")
    parser.add_argument("--input", default="input/example.jpg", help="Path to input image")
    parser.add_argument("--output", default="output/result.jpg", help="Path to save output image")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .tflite model")
    parser.add_argument("--labels", default=None, help="Path to labels.txt (default: alongside the model)")
    parser.add_argument("--no-delegate", action="store_true", help="Run inference without VX delegate (CPU only)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.labels is None:
        args.labels = os.path.join(os.path.dirname(args.model), "labels.txt")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cv_image = cv2.imread(args.input)
    if cv_image is None:
        sys.exit(f"Could not read image: {args.input}")

    labels = load_labels(args.labels)

    interpreter = load_interpreter(args.model, use_delegate=not args.no_delegate)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]
    input_dtype = input_details[0]['dtype']

    # Warmup: first invoke compiles delegate kernels; feed zeros so the timing is meaningful.
    zeros = np.zeros(input_details[0]['shape'], dtype=input_dtype)
    interpreter.set_tensor(input_details[0]['index'], zeros)
    warmup_start = time.time()
    interpreter.invoke()
    print(f"Interpreter warmup time: {time.time() - warmup_start:.2f} sec")

    input_data = preprocess(cv_image, input_height, input_width, input_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    inference_start = time.time()
    interpreter.invoke()
    inference_elapsed = time.time() - inference_start

    boxes, classes, scores = parse_ssd_outputs(interpreter, output_details)

    detections = []
    for idx, class_id in enumerate(classes):
        if scores[idx] > CONFIDENCE_THRESHOLD:
            label = labels.get(int(class_id), str(int(class_id)))
            detections.append((label, scores[idx], boxes[idx]))
            print(f"{label}: {scores[idx]:.2f}  bbox={boxes[idx].tolist()}")

    frame = draw_bounding_box(cv_image, detections)
    cv2.imwrite(args.output, frame)
    print(f"Inference complete in {inference_elapsed:.3f} sec. Output saved at {args.output}")


if __name__ == "__main__":
    main()
