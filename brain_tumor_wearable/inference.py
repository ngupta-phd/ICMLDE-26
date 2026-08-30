"""Compatibility wrapper for the original Streamlit app (`app.py`).

New code lives in `src/`. This module keeps `load_model`, `analyze_mri`,
and `publish_mqtt_result` so existing callers still work.
"""

import time

import numpy as np
from PIL import Image

from config import GRADCAM_OUTPUT_DIR
from src.gradcam import generate_gradcam
from src.inference import load_model, predict
from src.mqtt_publisher import publish_diagnosis

__all__ = ["load_model", "predict", "analyze_mri", "publish_mqtt_result"]


def analyze_mri(pil_image, model, target_layer_name="features.12"):
    """Run classification + Grad-CAM for a PIL image (used by app.py)."""
    del target_layer_name  # target is always the last MobileNetV3 feature block

    temp_path = GRADCAM_OUTPUT_DIR / "_streamlit_input.jpg"
    GRADCAM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pil_image.convert("RGB").save(temp_path)

    result = predict(temp_path, model=model)

    cam_start = time.perf_counter()
    cam_path = generate_gradcam(temp_path, model, class_idx=result["predicted_idx"])
    cam_time_ms = (time.perf_counter() - cam_start) * 1000.0
    cam_overlay = np.array(Image.open(cam_path).convert("RGB"))

    return {
        "pred_class": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "inf_time_ms": result["inference_time_ms"],
        "cam_time_ms": cam_time_ms,
        "cam_overlay": cam_overlay,
        "cam_path": cam_path,
    }


def publish_mqtt_result(result, image_name="uploaded_mri.jpg"):
    payload = {
        "image_name": image_name,
        "prediction": result["pred_class"],
        "confidence": round(result["confidence"], 2),
        "inference_time_ms": round(result["inf_time_ms"], 2),
        "gradcam_time_ms": round(result["cam_time_ms"], 2),
    }
    return publish_diagnosis(payload)
