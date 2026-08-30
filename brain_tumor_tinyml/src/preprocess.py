"""MRI image loading and MobileNetV3-compatible preprocessing."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import config


def image_transform(image_size: tuple[int, int] = config.IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(config.NORMALIZE_MEAN), std=list(config.NORMALIZE_STD)),
        ]
    )


def load_image_tensor(image_path: Path | str, image_size: tuple[int, int] = config.IMAGE_SIZE) -> torch.Tensor:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    image = Image.open(path).convert("RGB")
    tensor = image_transform(image_size)(image)
    return tensor.unsqueeze(0)
