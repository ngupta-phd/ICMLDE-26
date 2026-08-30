"""MobileNetV3-Small MRI classification (4 classes)."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from config import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_PATH


def get_device():
    """Use CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preprocess():
    """Resize to 224x224, tensor, ImageNet normalization."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def build_model(num_classes=None):
    """Create MobileNetV3-Small with a 4-class classifier head."""
    if num_classes is None:
        num_classes = len(CLASS_NAMES)

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def load_model(weights_path=None, device=None):
    """
    Load MobileNetV3-Small weights.

    Accepts a raw state_dict or a checkpoint dict with a 'state_dict' key.
    Strips DataParallel 'module.' prefixes when present.
    """
    if weights_path is None:
        weights_path = MODEL_PATH
    weights_path = Path(weights_path)

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {weights_path}\n"
            "Copy the trained weights from Colab to model/best_mobilenetv3.pth"
        )

    if device is None:
        device = get_device()

    model = build_model()

    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=device)

    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(image_path, model=None):
    """
    Run 4-class MRI classification on an image file.

    Returns a dict with prediction, confidence (%), class probabilities,
    inference time (ms), class index, and the device used.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}\n"
            "Pass a valid path, for example: python main.py mri_scan.jpg"
        )

    if model is None:
        model = load_model()

    device = next(model.parameters()).device
    preprocess = get_preprocess()

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
    inference_time_ms = (time.perf_counter() - start) * 1000.0

    predicted_idx = int(torch.argmax(probabilities).item())
    probs_list = probabilities.tolist()

    return {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "prediction": CLASS_NAMES[predicted_idx],
        "predicted_idx": predicted_idx,
        "confidence": float(probs_list[predicted_idx] * 100.0),
        "probabilities": {
            name: float(prob) for name, prob in zip(CLASS_NAMES, probs_list)
        },
        "inference_time_ms": inference_time_ms,
        "device": str(device),
    }
