U-Net checkpoint
================

File:    models/unet_brain_mri.pth
Format:  PyTorch state_dict (OrderedDict of tensors)
Size:    380,397 bytes (~371 KB)

Architecture: LightweightUNet (models/unet.py)
  encoder (3→32) → max-pool → bottleneck (32→64) → transpose upsample
  → skip-concat decoder (64→32→1) → sigmoid

Input:   N × 3 × 224 × 224  (RGB, values in [0, 1], bilinear resize)
Output:  N × 1 × 224 × 224  sigmoid probabilities
Type:    binary segmentation (background vs foreground)
Threshold used at inference: 0.5
Class mapping: 0 = background, 1 = predicted foreground (lesion/structure)

Checkpoint keys match LightweightUNet with no "module." prefix.
No training script is included in this repository; preprocessing follows the
existing inference code (RGB conversion, 224×224, divide by 255). There is no
ImageNet mean/std normalization.

Training dataset and metrics are not documented in-repo. Do not invent accuracy.
Do not replace this file with an unrelated architecture.
