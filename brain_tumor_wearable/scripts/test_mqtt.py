"""Publish a sample diagnosis message to MQTT.

Usage:
    python scripts/test_mqtt.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC
from src.mqtt_publisher import publish_diagnosis


def main():
    payload = {
        "image_name": "mri_scan.jpg",
        "prediction": "Glioma",
        "confidence": 94.32,
        "inference_time_ms": 28.4,
        "gradcam_time_ms": 15.7,
    }

    print(f"Publishing test payload to {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Topic: {MQTT_TOPIC}")
    print(payload)

    ok = publish_diagnosis(payload)
    if ok:
        print("Test MQTT publish succeeded.")
        return 0

    print("Test MQTT publish failed. Is a broker running on localhost:1883?")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
