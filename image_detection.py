# Copyright (c) 2025 cubesys GmbH
# Licensed under the MIT License. See LICENSE for details.

"""Single-image object detection on cube:evk; writes an annotated copy to disk."""

import argparse
import os
import sys

import cv2

from detector import ROAD_USER_LABELS, Detector, draw_bounding_box, merge_rider_pairs

DEFAULT_MODEL = "models/ssd_mobilenet_v1_1/ssd_mobilenet_v1_1.tflite"


def parse_args():
    parser = argparse.ArgumentParser(description="TFLite object detection on cube:evk")
    parser.add_argument("--input", default="input/image2.jpg", help="Path to input image")
    parser.add_argument("--output", default="output/result.jpg", help="Path to save output image")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .tflite model")
    parser.add_argument("--labels", default=None, help="Path to labels.txt (default: labels.txt next to the model)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--no-delegate", action="store_true", help="Disable VX delegate (CPU only)")
    parser.add_argument(
        "--all-labels",
        action="store_true",
        help=f"Detect all model classes (default keeps only road users: {', '.join(ROAD_USER_LABELS)})",
    )
    parser.add_argument(
        "--no-merge-riders",
        action="store_true",
        help="Disable merging overlapping person+bicycle / person+motorcycle into cyclist/motorcyclist",
    )
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

    detector = Detector(
        model_path=args.model,
        labels_path=args.labels,
        use_delegate=not args.no_delegate,
        confidence_threshold=args.threshold,
        allowed_labels=None if args.all_labels else ROAD_USER_LABELS,
    )
    print(f"Interpreter warmup time: {detector.warmup_time:.2f} sec")

    detections, inference_time = detector.detect(cv_image)
    if not args.no_merge_riders:
        detections = merge_rider_pairs(detections)
    for d in detections:
        print(f"{d.label}: {d.score:.2f}  bbox={list(d.box)}")

    frame = draw_bounding_box(cv_image, detections)
    cv2.imwrite(args.output, frame)
    print(f"Inference complete in {inference_time:.3f} sec. Output saved at {args.output}")


if __name__ == "__main__":
    main()
