"""Test MobileNetV3 inference and Grad-CAM (no MQTT).

Usage:
    python scripts/test_inference.py path/to/mri_scan.jpg
    python scripts/test_inference.py
"""

import sys
from pathlib import Path

# Allow running this file as: python scripts/test_inference.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_NAMES
from src.gradcam import generate_gradcam
from src.inference import load_model, predict


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) >= 1:
        image_path = Path(argv[0])
    else:
        image_path = PROJECT_ROOT / "mri_scan.jpg"

    print("=" * 55)
    print("BRAIN TUMOR MOBILENETV3 INFERENCE TEST")
    print("=" * 55)
    print()
    print("Image:")
    print(f"  {image_path}")
    print()

    try:
        model = load_model()
        result = predict(image_path, model=model)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    except Exception as exc:
        print(f"Inference failed: {exc}")
        return 1

    print("Prediction:")
    print(f"  {result['prediction']}")
    print()
    print("Confidence:")
    print(f"  {result['confidence']:.2f}%")
    print()
    print("Class probabilities:")
    for name in CLASS_NAMES:
        print(f"  {name}:")
        print(f"    {result['probabilities'][name] * 100:.2f}%")
    print()
    print("Inference time:")
    print(f"  {result['inference_time_ms']:.2f} ms")
    print()

    try:
        gradcam_path = generate_gradcam(
            image_path, model, class_idx=result["predicted_idx"]
        )
        print("Grad-CAM:")
        print(f"  {gradcam_path}")
    except Exception as exc:
        print("Grad-CAM:")
        print(f"  failed ({exc})")
        return 1

    print()
    print("=" * 55)
    print("Experimental prototype only — not a medical diagnosis.")
    print("=" * 55)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
