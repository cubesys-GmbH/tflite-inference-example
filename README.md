# TFLite Inference Example - cube:evk

Example project demonstrating image inference with bounding boxes using LiteRT (prev. TensorFlow Lite) on the **cube:evk**.

## Installation and Setup

### Clone the repository

```bash
git clone https://github.com/cubesys-GmbH/tflite-inference-example.git
cd tflite-inference-example
```

### Set up a Python virtual environment

It's recommended to use a virtual environment to isolate project dependencies.

#### On cube:evk
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Run It

```bash
python image_detection.py 
```


## License

This project is licensed under the MIT License. See LICENSE for details.

## Contribution & Support

Contributions welcome — open a PR or issue. 
