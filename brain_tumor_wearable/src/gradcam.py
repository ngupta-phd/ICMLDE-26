"""Simple Grad-CAM heatmap for MobileNetV3-Small."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from config import GRADCAM_OUTPUT_DIR
from src.inference import get_preprocess


def _target_layer(model):
    """Last convolutional block of MobileNetV3-Small (`features`)."""
    return model.features[-1]


class _GradCAMHooks:
    def __init__(self, layer):
        self.activations = None
        self.gradients = None
        self._handles = [
            layer.register_forward_hook(self._on_forward),
            layer.register_full_backward_hook(self._on_backward),
        ]

    def _on_forward(self, module, inputs, output):
        self.activations = output.detach()

    def _on_backward(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        for handle in self._handles:
            handle.remove()


def generate_gradcam(image_path, model, class_idx, output_dir=None):
    """
    Create a Grad-CAM overlay for the predicted class and save it.

    Returns the saved image path.
    """
    image_path = Path(image_path)
    if output_dir is None:
        output_dir = GRADCAM_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = Image.open(image_path).convert("RGB")
    orig_np = np.array(original)
    device = next(model.parameters()).device

    preprocess = get_preprocess()
    input_tensor = preprocess(original).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    layer = _target_layer(model)
    hooks = _GradCAMHooks(layer)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        score = logits[0, class_idx]
        score.backward()

        if hooks.activations is None or hooks.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        gradients = hooks.gradients[0]
        activations = hooks.activations[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        heatmap = cam.cpu().numpy()
    finally:
        hooks.remove()
        model.zero_grad(set_to_none=True)

    heatmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = np.clip(0.4 * heatmap_color + 0.6 * orig_np, 0, 255).astype(np.uint8)

    save_path = output_dir / f"{image_path.stem}_gradcam.jpg"
    Image.fromarray(overlay).save(save_path)
    return str(save_path)
