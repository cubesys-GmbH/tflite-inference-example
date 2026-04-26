# Copyright (c) 2025 cubesys GmbH
# Licensed under the MIT License. See LICENSE for details.

"""Live USB-camera object detection on cube:evk, served as MJPEG over HTTP.

The board is typically headless, so frames are streamed over the network.
View at http://<board-ip>:<port>/ from any browser on the LAN, or via an
SSH tunnel: `ssh -L 8080:localhost:8080 user@cube-evk` then open
http://localhost:8080/.
"""

import argparse
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from detector import Detector, draw_detections

DEFAULT_MODEL = "models/ssd_mobilenet_v1_1/ssd_mobilenet_v1_1.tflite"

INDEX_HTML = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>cube:evk live detection</title></head>
<body style="margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center">
<h2 style="padding:8px;margin:0">cube:evk live detection</h2>
<img src="/stream.mjpg" style="max-width:100%;height:auto;display:block;margin:8px auto"/>
</body></html>"""


class FrameBroker:
    """Holds the most recent JPEG frame; wakes waiting clients on update."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None

    def publish(self, jpeg_bytes: bytes) -> None:
        with self._cond:
            self._frame = jpeg_bytes
            self._cond.notify_all()

    def wait_next(self, timeout: float = 2.0):
        with self._cond:
            self._cond.wait(timeout=timeout)
            return self._frame


def make_handler(broker: FrameBroker):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args, **_kwargs):
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(INDEX_HTML)))
                self.end_headers()
                self.wfile.write(INDEX_HTML)
                return

            if self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpeg = broker.wait_next()
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
                return

            self.send_error(404)

    return Handler


def open_capture(device: str, width: int, height: int) -> cv2.VideoCapture:
    src = int(device) if device.isdigit() else device
    # Pin the V4L2 backend; the auto-selected backend in opencv-python-headless
    # frequently fails on integer indices even when /dev/videoN opens fine.
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened() and isinstance(src, int):
        cap = cv2.VideoCapture(f"/dev/video{src}", cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"Could not open video device: {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def parse_args():
    parser = argparse.ArgumentParser(description="Live USB-cam object detection on cube:evk (MJPEG over HTTP)")
    parser.add_argument("--device", default="0", help="V4L2 device index (e.g. 0) or path (e.g. /dev/video0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--bind", default="0.0.0.0", help="HTTP bind address (use 127.0.0.1 to require SSH tunnel)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .tflite model")
    parser.add_argument("--labels", default=None, help="Path to labels.txt (default: alongside the model)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--no-delegate", action="store_true", help="Disable VX delegate (CPU only)")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-100")
    return parser.parse_args()


def _print_access_hint(bind: str, port: int) -> None:
    if bind in ("0.0.0.0", "::"):
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            ip = "<board-ip>"
        print(f"Streaming at http://{ip}:{port}/  (LAN access)")
    else:
        print(f"Streaming at http://{bind}:{port}/")
    print(f"SSH tunnel from a remote host: ssh -L {port}:localhost:{port} <user>@<board>  then open http://localhost:{port}/")


def main():
    args = parse_args()
    if args.labels is None:
        args.labels = os.path.join(os.path.dirname(args.model), "labels.txt")

    detector = Detector(
        model_path=args.model,
        labels_path=args.labels,
        use_delegate=not args.no_delegate,
        confidence_threshold=args.threshold,
    )
    print(f"Interpreter warmup time: {detector.warmup_time:.2f} sec")

    cap = open_capture(args.device, args.width, args.height)
    broker = FrameBroker()

    server = ThreadingHTTPServer((args.bind, args.port), make_handler(broker))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _print_access_hint(args.bind, args.port)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, args.jpeg_quality))]
    frames = 0
    last_log = time.time()
    last_inference_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            detections, last_inference_time = detector.detect(frame)
            annotated = draw_detections(frame, detections)
            ok, buf = cv2.imencode(".jpg", annotated, encode_params)
            if ok:
                broker.publish(buf.tobytes())

            frames += 1
            now = time.time()
            if now - last_log >= 2.0:
                fps = frames / (now - last_log)
                print(f"~{fps:.1f} fps  inference={last_inference_time*1000:.1f} ms  detections={len(detections)}")
                frames = 0
                last_log = now
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        server.shutdown()


if __name__ == "__main__":
    main()
