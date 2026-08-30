"""Paths and inference constants for the trained LightweightUNet."""

import os

from models.unet import IMAGE_SIZE

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "unet_brain_mri.pth")
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "sample_data", "test_mri.jpg")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Binary sigmoid threshold used at inference (matches existing pipeline).
MASK_THRESHOLD = 0.5

# Spatial size used by existing inference; the network is fully convolutional.
INFERENCE_SIZE = IMAGE_SIZE
