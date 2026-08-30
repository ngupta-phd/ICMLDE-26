# XAI U-Net Brain MRI Edge Inference

Research/educational prototype that runs a small U-Net on a 2D brain MRI slice, writes a segmentation mask and visual explanation, and measures local inference time. **Not a medical diagnostic system.**

## Purpose

Brain MRI → U-Net inference → segmentation mask → overlay and probability map → CPU/edge-oriented output.

## Pipeline

```
MRI image
  → RGB conversion, bilinear resize to 224×224, scale to [0, 1]
  → LightweightUNet (eval, inference_mode)
  → sigmoid probability map
  → binary mask (threshold 0.5)
  → original | mask | overlay | probability heatmap
```

## U-Net

| Item | Value |
| --- | --- |
| Architecture | `LightweightUNet` in `models/unet.py` |
| Checkpoint | `models/unet_brain_mri.pth` (raw PyTorch `state_dict`, 380,397 bytes) |
| Input | `1 × 3 × 224 × 224` (batch, RGB, H, W), values in `[0, 1]` |
| Output | `1 × 1 × 224 × 224` sigmoid probabilities |
| Segmentation | Binary (background vs foreground) |
| Threshold | `0.5` |
| Parameters | 92,737 |
| Device | CPU by default with `--cpu`; otherwise CUDA if `torch.cuda.is_available()` |

The first convolution is `Conv2d(3, 32, …)` and the last is `Conv2d(32, 1, …)` followed by `Sigmoid`. Checkpoint keys match this module (no `module.` prefix).

No training code is in this repository. Preprocessing matches the existing inference path: PIL RGB, 224×224 bilinear resize, `/255`. There is **no** ImageNet mean/std normalization.

## XAI / visual explanation

The primary explanation is the **U-Net’s own output**:

- `segmentation_mask.png` — binary mask after threshold 0.5
- `segmentation_overlay.png` — MRI with the mask highlighted
- `probability_map.png` — sigmoid confidence heatmap overlaid on the MRI

These maps show where the network assigned high foreground probability. They do **not** prove medical correctness.

Optional `--gradcam` adds a bottleneck Grad-CAM overlay (`xai_gradcam.png`). It is a model-internal attribution map, not a clinical explanation, and it requires a backward pass so it is off by default for edge timing.

## Edge inference

- Shallow 32/64-channel U-Net (~93k parameters)
- `model.eval()` and `torch.inference_mode()` for the timed forward pass
- No ONNX/TFLite conversion (not required for a working CPU path)
- Timing is **local CPU (or CUDA if present)** — not a claim about a specific edge SoC unless you run on that hardware

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS: `source venv/bin/activate`.

Dependencies: `torch`, `numpy`, `opencv-python`, `Pillow`.

## Usage

From the project root:

```bash
python main.py --cpu
```

Default input: `sample_data/test_mri.jpg`. Outputs go to `outputs/`.

```bash
python main.py --image path\to\mri.jpg --output-dir outputs --cpu
python main.py --image path\to\mri.jpg --gradcam
```

If the checkpoint or image is missing, or the image/checkpoint cannot be read, the program exits with a clear error and does not invent a mask.

## Testing

```bash
python -m py_compile main.py pipeline.py config.py models\unet.py models\gradcam.py scripts\test_inference.py tests\test_pipeline.py
python -m unittest tests.test_pipeline -v
python scripts\test_inference.py
```

`scripts/test_inference.py` forces CPU, loads the checkpoint, runs the sample MRI, and prints shapes, mask stats, and inference time.

## Example output files

| File | Content |
| --- | --- |
| `outputs/original_mri.png` | Resized MRI |
| `outputs/segmentation_mask.png` | Binary mask |
| `outputs/segmentation_overlay.png` | MRI + mask |
| `outputs/probability_map.png` | Probability heatmap on MRI |
| `outputs/pipeline_panel.png` | Four-panel summary |

Generated files under `outputs/` are gitignored (folder kept with `.gitkeep`). All saved images are 224×224 (the U-Net input size).

## Project structure

```
.
├── main.py
├── pipeline.py
├── config.py
├── requirements.txt
├── models/
│   ├── unet.py
│   ├── gradcam.py
│   ├── unet_brain_mri.pth
│   └── README.txt
├── sample_data/
│   └── test_mri.jpg
├── scripts/
│   └── test_inference.py
├── tests/
│   └── test_pipeline.py
└── outputs/
```

`sample_data/test_mri.jpg` is a small demo slice. Large training datasets are not included.

## Limitations

- Research/educational prototype — not a medical diagnostic system.
- 2D slice inference only; not a 3D volume pipeline.
- No published in-repo Dice/IoU/accuracy; do not treat demo overlays as clinical validation.
- Foreground class meaning is whatever the checkpoint learned; this repo has no training labels. On `sample_data/test_mri.jpg`, the mask is a thin peripheral contour (skull/brain boundary), not the bright interior lesion. That is observed checkpoint behavior, not a tumor Dice score.
- Grad-CAM (optional) is attribution, not a validated medical explanation.
- Tested as **local CPU inference**. Deployment on a named edge device is untested unless you run it there.
- No license has been specified.

## Dataset

No large dataset is shipped. Pass additional MRI images with `--image`. Do not commit large evaluation/training sets.
