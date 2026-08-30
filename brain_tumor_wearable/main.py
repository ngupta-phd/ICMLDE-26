"""Run MRI classification, Grad-CAM, and optional MQTT publish.

Usage:
    python main.py path/to/mri_scan.jpg
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CLASS_NAMES, MODEL_PATH, MQTT_BROKER, MQTT_PORT, MQTT_TOPIC
from src.gradcam import generate_gradcam
from src.inference import load_model, predict
from src.mqtt_publisher import publish_diagnosis


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 1:
        print("Usage: python main.py path/to/mri_scan.jpg")
        print("This is an experimental MRI classification prototype, not a medical diagnosis.")
        return 1

    image_path = Path(argv[0])

    print("Loading model...")
    try:
        model = load_model()
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print(f"Loaded: {MODEL_PATH}")
    print("Running inference...")

    try:
        result = predict(image_path, model=model)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    except Exception as exc:
        print(f"Inference failed: {exc}")
        return 1

    print()
    print(f"Prediction:      {result['prediction']}")
    print(f"Confidence:      {result['confidence']:.2f}%")
    print(f"Inference time:  {result['inference_time_ms']:.2f} ms")
    print(f"Device:          {result['device']}")
    print()
    print("Class probabilities:")
    for name in CLASS_NAMES:
        print(f"  {name}: {result['probabilities'][name] * 100:.2f}%")

    print()
    print("Generating Grad-CAM...")
    cam_start = time.perf_counter()
    try:
        gradcam_path = generate_gradcam(
            image_path, model, class_idx=result["predicted_idx"]
        )
        gradcam_time_ms = (time.perf_counter() - cam_start) * 1000.0
        print(f"Grad-CAM saved:  {gradcam_path}")
        print(f"Grad-CAM time:   {gradcam_time_ms:.2f} ms")
    except Exception as exc:
        gradcam_path = None
        gradcam_time_ms = None
        print(f"Warning: Grad-CAM failed ({exc}). Continuing without visualization.")

    payload = {
        "image_name": result["image_name"],
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 2),
        "inference_time_ms": round(result["inference_time_ms"], 2),
        "gradcam_time_ms": (
            None if gradcam_time_ms is None else round(gradcam_time_ms, 2)
        ),
    }

    print()
    print(f"Publishing MQTT to {MQTT_BROKER}:{MQTT_PORT} topic '{MQTT_TOPIC}'...")
    ok = publish_diagnosis(payload)
    if ok:
        print("MQTT publish succeeded.")
    else:
        print("MQTT skipped or failed; local results above are still valid.")

    print()
    print("Done. This output is for research/demo only, not a clinical diagnosis.")
    if gradcam_path:
        print(f"Open Grad-CAM: {gradcam_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
