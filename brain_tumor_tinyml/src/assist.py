"""Map classifier output to a simple robotics assist recommendation.

This is decision support for a research prototype, not a medical device.
It does not command treatment.
"""

from __future__ import annotations

from dataclasses import dataclass

import config

# Simulated controller verbs a robot stack could consume.
ASSIST_CONTINUE = "CONTINUE"
ASSIST_SLOW_AND_FLAG = "SLOW_AND_FLAG"
ASSIST_HOLD_FOR_REVIEW = "HOLD_FOR_REVIEW"

_TUMOR_CLASSES = frozenset({"glioma", "meningioma", "pituitary"})


@dataclass(frozen=True)
class AssistDecision:
    condition: str
    class_index: int
    confidence: float
    scores: dict[str, float]
    assist_state: str
    message: str
    inference_ms: float
    tumor_area_pct: float | None = None


def decide_assist(
    class_index: int,
    confidence: float,
    scores: dict[str, float],
    inference_ms: float,
    tumor_area_pct: float | None = None,
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
) -> AssistDecision:
    if class_index < 0 or class_index >= len(config.CLASS_NAMES):
        raise ValueError(f"class_index {class_index} is outside CLASS_NAMES")

    condition = config.CLASS_NAMES[class_index]

    if confidence < confidence_threshold:
        state = ASSIST_HOLD_FOR_REVIEW
        message = (
            f"Low confidence ({confidence:.1%}) for '{condition}'. "
            "Hold motion and request operator review."
        )
    elif condition == "no_tumor":
        state = ASSIST_CONTINUE
        message = "Predicted no_tumor. Maintain planned trajectory; no tumor-avoidance overlay."
    elif condition in _TUMOR_CLASSES:
        state = ASSIST_SLOW_AND_FLAG
        extra = ""
        if tumor_area_pct is not None:
            extra = f" Optional mask coverage {tumor_area_pct:.1f}% of the frame."
        message = (
            f"Predicted {condition}. Slow approach and flag the region for operator review.{extra}"
        )
    else:
        state = ASSIST_HOLD_FOR_REVIEW
        message = f"Unhandled class '{condition}'. Hold for operator review."

    return AssistDecision(
        condition=condition,
        class_index=class_index,
        confidence=confidence,
        scores=scores,
        assist_state=state,
        message=message,
        inference_ms=inference_ms,
        tumor_area_pct=tumor_area_pct,
    )
