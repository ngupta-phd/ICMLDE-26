"""Subscribe to diagnosis telemetry and print it.

Usage:
    python mqtt_listener.py
"""

import json
import sys
from pathlib import Path

import paho.mqtt.client as mqtt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC


def _fmt(value, suffix=""):
    if value is None:
        return "(missing)"
    return f"{value}{suffix}"


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print(f"[MQTT Listener] Connected to {MQTT_BROKER}:{MQTT_PORT}")
        print(f"[MQTT Listener] Subscribed to '{MQTT_TOPIC}'")
        print("Waiting for messages. Press Ctrl+C to stop.\n")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"[MQTT Listener] Connection failed with code {reason_code}")


def on_message(client, userdata, msg):
    raw = msg.payload.decode(errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        print("[Error] Malformed JSON on topic", msg.topic)
        print(f"        {err}")
        print(f"        Payload: {raw[:200]}")
        print()
        return

    if not isinstance(data, dict):
        print("[Error] JSON payload is not an object.")
        print(f"        Payload: {raw[:200]}")
        print()
        return

    print("=" * 45)
    print("         NEW DIAGNOSIS TELEMETRY RECEIVED")
    print("=" * 45)
    print(f" Image Target     : {_fmt(data.get('image_name'))}")
    print(f" Predicted Class  : {_fmt(data.get('prediction'))}")
    print(f" Confidence Score : {_fmt(data.get('confidence'), '%')}")
    print(f" Inference Time   : {_fmt(data.get('inference_time_ms'), ' ms')}")
    print(f" Grad-CAM Compute : {_fmt(data.get('gradcam_time_ms'), ' ms')}")
    print("=" * 45 + "\n")


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopping MQTT listener.")
        client.disconnect()
    except Exception as exc:
        print(f"Could not connect to MQTT broker {MQTT_BROKER}:{MQTT_PORT}")
        print(f"Reason: {exc}")
        print("Start a local broker (for example Mosquitto) and try again.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
