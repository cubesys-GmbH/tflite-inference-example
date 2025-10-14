# TFLite Inference Example - cube:evk

Example project demonstrating image inference with bounding boxes using LiteRT (formerly TensorFlow Lite) on the **cube:evk**.

## AI on Edge with cube:evk

<p align="center">
  <img src="docs/cube-evk.png" alt="cube:evk" width="300">
</p>

The cube:evk is built around the i.MX8MPlus, a quad-core Arm® Cortex®-A53 applications processor running at up to 1.8 GHz. It integrates a neural processing unit (NPU) capable of delivering up to 2.3 TOPS, making it the first i.MX processor with a dedicated machine learning accelerator. This architecture enables significantly enhanced performance for ML inference at the edge.

In addition to its AI capabilities, the cube:evk offers a rich set of connectivity interfaces, including V2X technologies such as DSRC and C-V2X, enabling seamless integration with intelligent transportation and connected mobility systems.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/cubesys-GmbH/tflite-inference-example.git
cd tflite-inference-example
```

### 2. Set up a Python virtual environment

It's recommended to use a virtual environment to isolate project dependencies.

#### On cube:evk
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run It

### Verify VX Delegate

Ensure the VX delegate library is available on your cube:evk:

```bash
/usr/lib/libvx_delegate.so
```

### Run the Example

#### With VX Delegate (NPU accelerated)

```bash
python image_detection.py --input input/example.jpg --output output/result.jpg
```

#### CPU Only (disable delegate)

```bash
python image_detection.py --input input/example.jpg --output output/result.jpg --no-delegate
```

#### Example Result

<p align="center">
  <img src="docs/result-image.jpg" alt="TFLite Inference Example Output" width="600">
</p>

## License

This project is licensed under the MIT License. See LICENSE for details.

## Contribution & Support

Contributions welcome — open a PR or issue. 
