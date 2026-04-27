# Copyright (c) 2025 cubesys GmbH
# Licensed under the MIT License. See LICENSE for details.

"""Bounding-box drawing helpers for cv2 images."""

from typing import List, Tuple

import cv2
import numpy as np

from .inference import Detection


def draw_bounding_box(
    cv_image: np.ndarray,
    detections: List[Detection],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes only (no labels). Returns a new image; input is not mutated."""
    image = cv_image.copy()
    h, w = image.shape[:2]
    for det in detections:
        y1, x1, y2, x2 = det.box
        x1 = max(0, min(w, int(x1 * w)))
        x2 = max(0, min(w, int(x2 * w)))
        y1 = max(0, min(h, int(y1 * h)))
        y2 = max(0, min(h, int(y2 * h)))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def draw_detections(
    cv_image: np.ndarray,
    detections: List[Detection],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes annotated with label and score (for live view). Input is not mutated."""
    image = cv_image.copy()
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for det in detections:
        y1, x1, y2, x2 = det.box
        x1 = max(0, min(w, int(x1 * w)))
        x2 = max(0, min(w, int(x2 * w)))
        y1 = max(0, min(h, int(y1 * h)))
        y2 = max(0, min(h, int(y2 * h)))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        text = f"{det.label} {det.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
        ty2 = y1 if y1 - th - baseline - 4 >= 0 else min(h, y1 + th + baseline + 4)
        ty1 = ty2 - th - baseline - 4
        cv2.rectangle(image, (x1, ty1), (min(w, x1 + tw + 4), ty2), color, -1)
        cv2.putText(image, text, (x1 + 2, ty2 - baseline - 2), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return image
