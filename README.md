# Edge AI with cube:evk

<p align="center">
  <img src="docs/cube-evk.png" alt="cube:evk" width="300">
</p>

The **cube:evk** is powered by an i.MX8M Plus quad-core Arm Cortex-A53 (up to 1.8 GHz) with an integrated NPU delivering up to **2.3 TOPS** for on-device AI inference. It also exposes V2X interfaces (DSRC, C-V2X) for connected edge applications.

More about cube:evk: [https://cubesys.io/#product-section](https://cubesys.io/#product-section)

## What This Example Does

Runs object detection on a single image using **SSD MobileNet v1** (quantized, ~4 MB, 80 COCO classes) via LiteRT (formerly TensorFlow Lite). The script:

1. Loads the model with the **VX delegate** so inference runs on the NPU (or falls back to CPU with `--no-delegate`).
2. Resizes the input image to the model's **300 × 300** input.
3. Runs inference and keeps detections with confidence **> 0.6**.
4. Draws green bounding boxes and writes the result to disk.
5. Prints warmup time and inference time to stdout.

Bundled artifacts:
- Model: `models/ssd_mobilenet_v1_1/ssd_mobilenet_v1_1.tflite`
- Labels: `models/ssd_mobilenet_v1_1/labels.txt` (80 COCO classes)
- Sample input: `input/example.jpg`

## Installation

Run these commands directly on the cube:evk.

### 1. Clone the repository

```bash
git clone https://github.com/cubesys-GmbH/tflite-inference-example.git
cd tflite-inference-example
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Installed packages: `numpy==1.26`, `opencv-python-headless`, `tflite-runtime`, `pillow`.

## Usage

### Verify the VX delegate is present

```bash
ls /usr/lib/libvx_delegate.so
```

If the file is missing, the script will fall back to CPU and print a warning.

### Run with NPU acceleration (default)

```bash
python image_detection.py --input input/example.jpg --output output/result.jpg
```

### Run on CPU only

```bash
python image_detection.py --input input/example.jpg --output output/result.jpg --no-delegate
```

### Arguments

| Flag            | Default                | Description                                       |
| --------------- | ---------------------- | ------------------------------------------------- |
| `--input`       | `input/example.jpg`    | Path to the input image.                          |
| `--output`      | `output/result.jpg`    | Path to write the annotated image (dir auto-created). |
| `--no-delegate` | *(off)*                | Disable the VX delegate; run inference on CPU.    |

### Expected output

```
VX delegate loaded (NPU acceleration enabled)
Interpreter warmup time: 0.XX sec
[(class_id, confidence, [y1, x1, y2, x2]), ...]
Inference complete in 0.XXX sec. Output saved at output/result.jpg
```

Compare warmup/inference times with and without `--no-delegate` to see the NPU speedup.

### Example result

<p align="center">
  <img src="docs/result-image.jpg" alt="TFLite Inference Example Output" width="600">
</p>

## License

MIT — see [LICENSE](./LICENSE).

## Contribution & Support

Contributions welcome — open a PR or issue.
