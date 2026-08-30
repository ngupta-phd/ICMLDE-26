"""Shared configuration for the brain tumor MRI prototype."""

from pathlib import Path

# Project root (directory that contains this file)
PROJECT_ROOT = Path(__file__).resolve().parent

# Trained MobileNetV3-Small checkpoint (copied in after Colab training)
MODEL_PATH = PROJECT_ROOT / "model" / "best_mobilenetv3.pth"

# Must match ImageFolder class order used during training (alphabetical folders)
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]

IMAGE_SIZE = 224

# ImageNet stats used by torchvision MobileNetV3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "brain_tumor/diagnosis"

GRADCAM_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"
