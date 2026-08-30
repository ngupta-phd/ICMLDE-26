# brain_tumor_wearable

Experimental local prototype: 4-class brain MRI classification with **MobileNetV3-Small**, optional Grad-CAM, and optional MQTT telemetry.

This is **not** a medical device and does **not** provide a clinical diagnosis.

## Classes

- Glioma
- Meningioma
- No Tumor
- Pituitary

## Directory structure

```
brain_tumor_wearable/
├── model/
│   └── best_mobilenetv3.pth   # add this after Colab training
├── src/
│   ├── inference.py
│   ├── gradcam.py
│   └── mqtt_publisher.py
├── scripts/
│   ├── test_inference.py
│   └── test_mqtt.py
├── outputs/
│   └── gradcam/
├── config.py
├── main.py
├── mqtt_listener.py
├── requirements.txt
└── README.md
```

## Installation

From the project root:

```text
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS use `source venv/bin/activate`.

## Put the trained model in `model/`

Train MobileNetV3-Small in Google Colab (same 4 classes, 224×224, ImageNet normalization). Then copy the checkpoint to:

```text
model/best_mobilenetv3.pth
```

The loader accepts either a raw `state_dict` or a dict with a `"state_dict"` key. `module.` prefixes are stripped if present.

Until that file exists, inference will print a clear error and exit.

## Run inference

```text
python main.py path/to/mri_scan.jpg
```

Example using the sample image in the project folder:

```text
python main.py mri_scan.jpg
```

Flow: load model → classify → print prediction/confidence/time → save Grad-CAM → try MQTT → exit.

If MQTT is down, you get a warning. Classification still completes.

## MQTT listener

Start a broker on `localhost:1883` (for example Mosquitto), then:

```text
python mqtt_listener.py
```

It subscribes to `brain_tumor/diagnosis` and prints telemetry. Malformed JSON and missing fields are handled without crashing.

## Test MQTT only

```text
python scripts/test_mqtt.py
```

## Test inference (no MQTT)

```text
python scripts/test_inference.py mri_scan.jpg
```

### Expected output

```text
=======================================================
BRAIN TUMOR MOBILENETV3 INFERENCE TEST

Image:
  ...

Prediction:
  ...

Confidence:
  ...%

Class probabilities:
  Glioma:
    ...
  Meningioma:
    ...
  No Tumor:
    ...
  Pituitary:
    ...

Inference time:
  ... ms

Grad-CAM:
  outputs/gradcam/..._gradcam.jpg
```

Settings (model path, classes, MQTT, image size) live in `config.py`.
