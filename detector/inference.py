# Copyright (c) 2025 cubesys GmbH
# Licensed under the MIT License. See LICENSE for details.

import os
import time
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

DEFAULT_VX_DELEGATE_PATH = "/usr/lib/libvx_delegate.so"


@dataclass
class Detection:
    label: str
    score: float
    box: Tuple[float, float, float, float]  # (y1, x1, y2, x2), normalized [0, 1]


class Detector:
    """SSD-MobileNet style object detector backed by a TFLite interpreter.

    Loads the model (optionally on the VX NPU delegate) and exposes a single
    `detect()` call that returns a list of Detection objects plus the
    inference time in seconds.
    """

    def __init__(
        self,
        model_path: str,
        labels_path: Optional[str] = None,
        use_delegate: bool = True,
        vx_delegate_path: str = DEFAULT_VX_DELEGATE_PATH,
        confidence_threshold: float = 0.6,
    ):
        self.confidence_threshold = confidence_threshold
        self.labels = self._load_labels(labels_path) if labels_path else {}

        delegates = []
        if not use_delegate:
            print("Running inference on CPU (delegate disabled)")
        elif not os.path.exists(vx_delegate_path):
            print(f"VX delegate not found at {vx_delegate_path}; running on CPU fallback")
        else:
            try:
                delegates.append(load_delegate(vx_delegate_path))
                print("VX delegate loaded (NPU acceleration enabled)")
            except Exception as e:
                print(f"Failed to load VX delegate ({e}); running on CPU fallback")

        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=delegates,
            num_threads=cpu_count(),
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self._input_shape = self.input_details[0]["shape"]
        self._input_dtype = self.input_details[0]["dtype"]

        zeros = np.zeros(self._input_shape, dtype=self._input_dtype)
        self.interpreter.set_tensor(self.input_details[0]["index"], zeros)
        warmup_start = time.time()
        self.interpreter.invoke()
        self.warmup_time = time.time() - warmup_start

    @staticmethod
    def _load_labels(path: str) -> dict:
        labels = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    labels[int(parts[0])] = parts[1]
        return labels

    def _preprocess(self, cv_image: np.ndarray) -> np.ndarray:
        height = self._input_shape[1]
        width = self._input_shape[2]
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height))
        batch = np.expand_dims(resized, axis=0)
        if self._input_dtype == np.float32:
            batch = (batch / 255.0).astype(np.float32)
        return batch

    def _parse_outputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Standard TFLite SSD postprocess names; positional fallback otherwise.
        by_name = {od["name"]: od for od in self.output_details}
        base = "TFLite_Detection_PostProcess"
        if base in by_name:
            boxes_d = by_name[base]
            classes_d = by_name[f"{base}:1"]
            scores_d = by_name[f"{base}:2"]
        else:
            boxes_d, classes_d, scores_d = self.output_details[:3]

        boxes = np.squeeze(self.interpreter.get_tensor(boxes_d["index"]))
        classes = np.squeeze(self.interpreter.get_tensor(classes_d["index"])).astype(int)
        scores = np.squeeze(self.interpreter.get_tensor(scores_d["index"]))
        return boxes, classes, scores

    def detect(self, cv_image: np.ndarray) -> Tuple[List[Detection], float]:
        """Run inference on a BGR image. Returns (detections, inference_time_seconds)."""
        input_data = self._preprocess(cv_image)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        start = time.time()
        self.interpreter.invoke()
        elapsed = time.time() - start

        boxes, classes, scores = self._parse_outputs()
        detections: List[Detection] = []
        for idx, class_id in enumerate(classes):
            if scores[idx] > self.confidence_threshold:
                label = self.labels.get(int(class_id), str(int(class_id)))
                detections.append(
                    Detection(label=label, score=float(scores[idx]), box=tuple(boxes[idx]))
                )
        return detections, elapsed
