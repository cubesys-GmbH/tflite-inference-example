# Copyright (c) 2025 cubesys GmbH
# Licensed under the MIT License. See LICENSE for details.

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
