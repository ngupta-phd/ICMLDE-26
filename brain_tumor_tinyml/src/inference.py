"""Classifier (and optional U-Net) inference with latency measurement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import config
from models.classifier import load_classifier
from models.unet import LightweightUNet, load_unet_model
from src.assist import AssistDecision, decide_assist
from src.preprocess import load_image_tensor


def softmax_scores(logits: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=1)[0]
    return {name: float(probs[i].item()) for i, name in enumerate(config.CLASS_NAMES)}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_classifier_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int = config.WARMUP_RUNS,
    runs: int = config.TIMING_RUNS,
) -> tuple[torch.Tensor, float]:
    """Return last logits and mean inference time in milliseconds."""
    import time

    model.eval()
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = model(input_tensor)
            _sync(device)

        times_ms: list[float] = []
        logits = None
        for _ in range(max(1, runs)):
            _sync(device)
            start = time.perf_counter()
            logits = model(input_tensor)
            _sync(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    assert logits is not None
    return logits, float(sum(times_ms) / len(times_ms))


def optional_tumor_area_pct(
    unet: LightweightUNet,
    input_tensor: torch.Tensor,
    threshold: float = config.SEGMENT_THRESHOLD,
) -> float:
    with torch.no_grad():
        mask = unet(input_tensor)
    binary = (mask.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    return float(binary.mean() * 100.0)


def decision_from_logits(
    logits: torch.Tensor,
    inference_ms: float,
    tumor_area_pct: float | None = None,
) -> AssistDecision:
    scores = softmax_scores(logits)
    class_index = int(torch.argmax(logits, dim=1).item())
    confidence = scores[config.CLASS_NAMES[class_index]]
    return decide_assist(
        class_index=class_index,
        confidence=confidence,
        scores=scores,
        inference_ms=inference_ms,
        tumor_area_pct=tumor_area_pct,
    )


def classify_image(
    image_path: Path | str,
    weights_path: Path | str | None = None,
    device: torch.device | str | None = None,
    segment: bool = False,
    unet_weights: Path | str | None = None,
    warmup: int = config.WARMUP_RUNS,
    runs: int = config.TIMING_RUNS,
    model: torch.nn.Module | None = None,
    unet: LightweightUNet | None = None,
) -> AssistDecision:
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    input_tensor = load_image_tensor(image_path).to(device)
    if model is None:
        model = load_classifier(weights_path=weights_path, device=device)

    logits, mean_ms = measure_classifier_latency(
        model, input_tensor, device, warmup=warmup, runs=runs
    )

    tumor_area_pct = None
    if segment:
        if unet is None:
            unet = load_unet_model(weights_path=unet_weights, device=device)
        tumor_area_pct = optional_tumor_area_pct(unet, input_tensor)

    return decision_from_logits(logits, mean_ms, tumor_area_pct)
