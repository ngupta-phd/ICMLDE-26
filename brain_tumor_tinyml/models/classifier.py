"""MobileNetV3-Small classifier for four MRI tumor categories."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

import config


def build_classifier(num_classes: int = 4) -> nn.Module:
    """Build MobileNetV3-Small with a 4-class head. No pretrained ImageNet fallback."""
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def load_classifier(weights_path: Path | str | None = None, device: torch.device | str = "cpu") -> nn.Module:
    weights_path = Path(weights_path) if weights_path else config.CLASSIFIER_WEIGHTS
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Classifier weights not found: {weights_path}\n"
            "Place best_mobilenetv3.pth in models/ or pass --weights."
        )

    model = build_classifier(num_classes=len(config.CLASS_NAMES))
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
