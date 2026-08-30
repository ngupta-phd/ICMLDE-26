# TinyML Medical Robotics Real-Time Assist

Research prototype that runs a **lightweight MRI tumor classifier on CPU** and maps the result to a **simple robotics assist recommendation**.

**Decision support / research prototype — not a medical device.** It does not diagnose, treat, or control clinical hardware.

## Purpose

Demonstrate an edge-oriented pipeline:

`MRI image → preprocess → MobileNetV3-Small inference → assist state + latency`

A software **mock** of a robot command interface formats the recommendation. No robot hardware is required or used.

## Architecture

```
sample MRI / camera-like loop
        │
        ▼
  preprocess (224×224, ImageNet normalize)
        │
        ▼
  MobileNetV3-Small (CPU)     ← actual ML inference
        │
        ▼
  assist mapping (CONTINUE / SLOW_AND_FLAG / HOLD_FOR_REVIEW)
        │
        ▼
  SimulatedRobotAssist (software mock only)
```

Optional `--segment` runs a small U-Net for mask coverage. That path is extra latency and is **not** the default real-time path.

## Model

| Item | Value |
| --- | --- |
| File | `models/best_mobilenetv3.pth` (~6.2 MB, included) |
| Framework | PyTorch |
| Architecture | torchvision `mobilenet_v3_small`, classifier head `Linear(1024, 4)` |
| Input | `1 × 3 × 224 × 224` float tensor, RGB, ImageNet mean/std |
| Output | 4 logits → softmax |
| Classes (index order) | `models/labels.txt`: `glioma`, `meningioma`, `no_tumor`, `pituitary` |

Class order is **not stored in the checkpoint**. It is read from `models/labels.txt`. Do not assume it matches an external dataset unless you trained with the same labels.

Optional localization (off by default):

| Item | Value |
| --- | --- |
| File | `models/unet_brain_mri.pth` (~380 KB, included) |
| Architecture | `LightweightUNet` in `models/unet.py` |
| Input | same tensor as the classifier |
| Output | `1 × 1 × 224 × 224` sigmoid mask |

## Assist output

| Predicted class | Confidence | Assist state | Meaning |
| --- | --- | --- | --- |
| any | `< 0.60` | `HOLD_FOR_REVIEW` | Uncertain; hold and ask an operator |
| `no_tumor` | `≥ 0.60` | `CONTINUE` | No tumor-avoidance overlay |
| `glioma` / `meningioma` / `pituitary` | `≥ 0.60` | `SLOW_AND_FLAG` | Slow approach; flag for review |

The mock robot maps those states to a speed scale (`1.0` / `0.4` / `0.0`). That is a **simulation**, not hardware.

## Install

Python 3.10+ recommended. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

CPU-only PyTorch is enough. Do not install `opencv-python` or `grad-cam` for this pipeline.

## Run

Default sample image:

```bash
python main.py
```

Custom image:

```bash
python main.py --image path/to/slice.jpg
```

Simulated real-time loop (same image, N frames; model loaded once):

```bash
python main.py --stream 20 --warmup 2 --runs 1
```

Optional U-Net coverage (not the default TinyML path):

```bash
python main.py --segment
```

## Expected output

Text similar to:

```
Decision support / research prototype — not a medical device.
Condition:        <one of glioma|meningioma|no_tumor|pituitary>
Confidence:        ...
Class scores:     glioma=... meningioma=... no_tumor=... pituitary=...
Assist state:     CONTINUE | SLOW_AND_FLAG | HOLD_FOR_REVIEW
Assist message:   ...
Inference time:   ... ms (CPU/edge-oriented measurement)
Simulated robot:  {'source': 'software_simulation', ...}
```

The predicted class on `sample_data/test_mri.jpg` depends on the shipped weights. Treat it as a demo input, not a labeled clinical case.

Missing image or missing `.pth` files print an error and exit with code `1`. The app does **not** invent predictions from random heads.

## Tests

```bash
python -m py_compile main.py config.py models/classifier.py models/unet.py src/preprocess.py src/inference.py src/assist.py src/robot_sim.py tests/test_pipeline.py
python -m unittest tests.test_pipeline -v
```

## Latency

`main.py` warms up, then averages `--runs` forward passes (`time.perf_counter` on CPU by default).

On the development machine used to prepare this repo (Windows, CPU, PyTorch): about **14.3 ms** mean over 10 timed runs after 2 warmup passes for the classifier on `sample_data/test_mri.jpg`. That is one local measurement, not a published benchmark. Re-run `python main.py` on your hardware.

## Limitations

- Not a medical device; not for clinical use.
- No robotics hardware in this repository; the robot interface is a mock.
- Class index mapping is project-defined, not embedded in the `.pth`.
- Input is a still image (or a repeated image loop), not a live surgical camera.
- MobileNetV3-Small in PyTorch is edge-oriented but not a microcontroller TFLite binary.
- No accuracy claims are made here; this repo ships inference, not a validation study.

## Layout

```
.
├── models/
│   ├── labels.txt
│   ├── best_mobilenetv3.pth
│   ├── unet_brain_mri.pth
│   ├── classifier.py
│   └── unet.py
├── src/
│   ├── preprocess.py
│   ├── inference.py
│   ├── assist.py
│   └── robot_sim.py      # simulation only
├── sample_data/
│   └── test_mri.jpg
├── tests/
│   └── test_pipeline.py
├── outputs/              # generated files (gitignored)
├── main.py
├── config.py
├── requirements.txt
└── README.md
```
