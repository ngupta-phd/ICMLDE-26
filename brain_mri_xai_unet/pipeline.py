"""U-Net brain MRI inference: preprocess → segment → visualize."""

import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from config import (
    CHECKPOINT_PATH,
    DEFAULT_OUTPUT_DIR,
    INFERENCE_SIZE,
    MASK_THRESHOLD,
)
from models.unet import LightweightUNet


class PipelineError(RuntimeError):
    """Raised when inference cannot proceed (missing files, bad checkpoint, bad image)."""


def select_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_unet(checkpoint_path=CHECKPOINT_PATH, device=None):
    if device is None:
        device = select_device()

    if not os.path.isfile(checkpoint_path):
        raise PipelineError(
            f"U-Net checkpoint not found: {checkpoint_path}\n"
            "Place the trained weights at models/unet_brain_mri.pth."
        )

    try:
        try:
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise PipelineError(
            f"Checkpoint could not be loaded (invalid or unreadable file): {checkpoint_path}"
        ) from exc

    if isinstance(state, dict):
        for wrap_key in ("state_dict", "model_state_dict", "model"):
            inner = state.get(wrap_key)
            if isinstance(inner, dict) and any(
                k.startswith("enc1.") or k.startswith("module.enc1.") for k in inner
            ):
                state = inner
                break

    if not isinstance(state, dict) or not state:
        raise PipelineError(
            f"Checkpoint format is invalid (expected a state_dict): {checkpoint_path}"
        )

    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}

    model = LightweightUNet()
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise PipelineError(
            "Checkpoint does not match LightweightUNet. "
            "Do not substitute an unrelated architecture."
        ) from exc

    model.to(device)
    model.eval()
    return model


def preprocess_mri(image_path, device):
    if not os.path.isfile(image_path):
        raise PipelineError(f"MRI image not found: {image_path}")

    try:
        image = Image.open(image_path)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise PipelineError(f"MRI image could not be opened: {image_path}") from exc

    width, height = image.size
    if width < 1 or height < 1:
        raise PipelineError(
            f"MRI image has invalid dimensions: {image.size} ({image_path})"
        )

    rgb = image.convert("RGB")
    resized = rgb.resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise PipelineError(
            f"Expected a 3-channel image after RGB conversion, got shape {array.shape}"
        )

    tensor = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
        .to(device)
    )
    return rgb, tensor


def run_inference(model, input_tensor):
    """Forward pass only. Returns probabilities (1, 1, H, W) and elapsed milliseconds."""
    device = input_tensor.device
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        probabilities = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
    return probabilities, elapsed_ms


def postprocess(probabilities, threshold=MASK_THRESHOLD):
    prob_map = probabilities.squeeze().detach().cpu().numpy().astype(np.float32)
    if prob_map.ndim != 2:
        raise PipelineError(
            f"Unexpected U-Net output shape after squeeze: {tuple(prob_map.shape)}"
        )
    mask = (prob_map > threshold).astype(np.uint8) * 255
    coverage_pct = float(np.mean(mask > 0) * 100.0)
    return prob_map, mask, coverage_pct


def _bgr_from_rgb(rgb_image):
    resized = np.array(
        rgb_image.resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.BILINEAR)
    )
    return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)


def _probability_heatmap(prob_map):
    heat = np.clip(prob_map, 0.0, 1.0)
    return cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET)


def _segmentation_overlay(mri_bgr, mask):
    highlight = np.zeros_like(mri_bgr)
    highlight[:, :, 2] = mask
    return cv2.addWeighted(mri_bgr, 0.65, highlight, 0.35, 0)


def save_visualizations(orig_rgb, mask, prob_map, output_dir, gradcam=None):
    os.makedirs(output_dir, exist_ok=True)
    mri_bgr = _bgr_from_rgb(orig_rgb)
    overlay = _segmentation_overlay(mri_bgr, mask)
    heatmap = _probability_heatmap(prob_map)
    heatmap_on_mri = cv2.addWeighted(mri_bgr, 0.55, heatmap, 0.45, 0)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    paths = {
        "original": os.path.join(output_dir, "original_mri.png"),
        "mask": os.path.join(output_dir, "segmentation_mask.png"),
        "overlay": os.path.join(output_dir, "segmentation_overlay.png"),
        "probability": os.path.join(output_dir, "probability_map.png"),
        "panel": os.path.join(output_dir, "pipeline_panel.png"),
    }

    cv2.imwrite(paths["original"], mri_bgr)
    cv2.imwrite(paths["mask"], mask)
    cv2.imwrite(paths["overlay"], overlay)
    cv2.imwrite(paths["probability"], heatmap_on_mri)
    cv2.imwrite(
        paths["panel"],
        np.hstack([mri_bgr, mask_bgr, overlay, heatmap_on_mri]),
    )

    if gradcam is not None:
        cam_color = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        cam_overlay = cv2.addWeighted(mri_bgr, 0.6, cam_color, 0.4, 0)
        paths["gradcam"] = os.path.join(output_dir, "xai_gradcam.png")
        cv2.imwrite(paths["gradcam"], cam_overlay)

    return paths


def maybe_gradcam(model, input_tensor):
    from models.gradcam import UNetGradCAM

    cam = UNetGradCAM(model)
    tensor = input_tensor.detach().clone().requires_grad_(True)
    try:
        _, heatmap, _ = cam.generate(tensor)
    finally:
        cam.close()
    return heatmap


def run_pipeline(
    image_path,
    output_dir=DEFAULT_OUTPUT_DIR,
    checkpoint_path=CHECKPOINT_PATH,
    force_cpu=False,
    with_gradcam=False,
    warmup=True,
):
    device = select_device(force_cpu=force_cpu)
    model = load_unet(checkpoint_path, device)
    orig_rgb, input_tensor = preprocess_mri(image_path, device)

    if warmup:
        with torch.inference_mode():
            _ = model(input_tensor)

    probabilities, elapsed_ms = run_inference(model, input_tensor)
    prob_map, mask, coverage_pct = postprocess(probabilities)

    gradcam = None
    if with_gradcam:
        gradcam = maybe_gradcam(model, input_tensor)

    paths = save_visualizations(orig_rgb, mask, prob_map, output_dir, gradcam=gradcam)
    n_params = sum(parameter.numel() for parameter in model.parameters())

    return {
        "device": str(device),
        "image_path": os.path.abspath(image_path),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "input_shape": tuple(input_tensor.shape),
        "output_shape": tuple(probabilities.shape),
        "mask_shape": tuple(mask.shape),
        "threshold": MASK_THRESHOLD,
        "coverage_pct": coverage_pct,
        "positive_pixels": int(np.sum(mask > 0)),
        "prob_min": float(prob_map.min()),
        "prob_max": float(prob_map.max()),
        "prob_mean": float(prob_map.mean()),
        "inference_ms": elapsed_ms,
        "parameter_count": n_params,
        "paths": paths,
        "warmup": warmup,
    }
