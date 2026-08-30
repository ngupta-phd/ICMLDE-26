"""Project paths and inference settings. Paths are relative to this file."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_SIZE = (224, 224)

# ImageNet-style normalization used with torchvision MobileNetV3.
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

CLASSIFIER_WEIGHTS = PROJECT_ROOT / "models" / "best_mobilenetv3.pth"
UNET_WEIGHTS = PROJECT_ROOT / "models" / "unet_brain_mri.pth"
LABELS_FILE = PROJECT_ROOT / "models" / "labels.txt"
SAMPLE_IMAGE = PROJECT_ROOT / "sample_data" / "test_mri.jpg"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _load_class_names() -> tuple[str, ...]:
    if not LABELS_FILE.is_file():
        raise FileNotFoundError(
            f"Class labels file not found: {LABELS_FILE}\n"
            "Expected one class name per line matching the 4-logit classifier head."
        )
    names = tuple(line.strip() for line in LABELS_FILE.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(names) != 4:
        raise ValueError(f"Expected 4 class names in {LABELS_FILE}, found {len(names)}")
    return names


# Index order for the 4 classifier logits. Not stored inside the .pth file.
CLASS_NAMES = _load_class_names()

# Below this softmax score the assist recommendation is HOLD_FOR_REVIEW.
CONFIDENCE_THRESHOLD = 0.60

# U-Net sigmoid threshold for optional localization mask.
SEGMENT_THRESHOLD = 0.5

WARMUP_RUNS = 2
TIMING_RUNS = 10
