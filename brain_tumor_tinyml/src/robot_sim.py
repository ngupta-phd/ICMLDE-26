"""Software mock of a robot command interface.

No hardware is attached. This only formats the assist decision as a command
a controller could consume. It is not a medical device.
"""

from __future__ import annotations

from src.assist import AssistDecision


class SimulatedRobotAssist:
    def apply(self, decision: AssistDecision) -> dict:
        command = {
            "source": "software_simulation",
            "command": decision.assist_state,
            "condition": decision.condition,
            "confidence": round(decision.confidence, 4),
            "max_speed_scale": _speed_scale(decision.assist_state),
            "operator_flag": decision.assist_state != "CONTINUE",
        }
        return command


def _speed_scale(assist_state: str) -> float:
    if assist_state == "CONTINUE":
        return 1.0
    if assist_state == "SLOW_AND_FLAG":
        return 0.4
    return 0.0


def format_report(decision: AssistDecision, robot_command: dict) -> str:
    scores = ", ".join(f"{name}={score:.3f}" for name, score in decision.scores.items())
    lines = [
        "Decision support / research prototype - not a medical device.",
        f"Condition:        {decision.condition}",
        f"Confidence:       {decision.confidence:.1%}",
        f"Class scores:     {scores}",
        f"Assist state:     {decision.assist_state}",
        f"Assist message:   {decision.message}",
        f"Inference time:   {decision.inference_ms:.2f} ms (CPU/edge-oriented measurement)",
        f"Simulated robot:  {robot_command}",
    ]
    if decision.tumor_area_pct is not None:
        lines.append(f"Mask coverage:    {decision.tumor_area_pct:.2f}% of frame (optional U-Net)")
    return "\n".join(lines)
