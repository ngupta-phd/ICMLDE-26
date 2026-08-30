"""Minimal MQTT publisher for diagnosis telemetry."""

import json

import paho.mqtt.client as mqtt

from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC


def publish_diagnosis(payload, broker=None, port=None, topic=None):
    """
    Connect, publish one JSON message, disconnect.

    Returns True on success, False if the broker is unavailable.
    """
    if broker is None:
        broker = MQTT_BROKER
    if port is None:
        port = MQTT_PORT
    if topic is None:
        topic = MQTT_TOPIC

    message = json.dumps(payload)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    try:
        client.connect(broker, port, keepalive=10)
        result = client.publish(topic, message, qos=0)
        result.wait_for_publish(timeout=5)
        client.disconnect()
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"Warning: MQTT publish failed with code {result.rc}.")
            return False
        return True
    except Exception as exc:
        print(f"Warning: MQTT unavailable ({exc}). Diagnosis still completed.")
        try:
            client.disconnect()
        except Exception:
            pass
        return False
