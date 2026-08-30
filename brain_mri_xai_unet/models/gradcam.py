"""Optional Grad-CAM on the U-Net bottleneck.

The primary explanation for this project is the segmentation probability map.
This module is only used when --gradcam is passed.
"""

import cv2
import numpy as np


class UNetGradCAM:
    """Grad-CAM on bottleneck conv, targeting the sum of the segmentation output."""

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = self.model.bottleneck[3]
        self._forward_handle = target_layer.register_forward_hook(self.save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, inputs, output):
        self.activations = output.clone()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone()

    def close(self):
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, input_tensor):
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()
        output_mask = self.model(input_tensor)

        target = output_mask.sum()
        target.backward()

        if self.gradients is None or self.activations is None:
            self.model.train(was_training)
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "The U-Net bottleneck target layer may be incompatible."
            )

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for index, weight in enumerate(weights):
            cam += weight * activations[index, :, :]

        height, width = output_mask.shape[-2:]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (width, height))
        cam -= np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        mask_array = (output_mask.squeeze().detach().cpu().numpy() > 0.5).astype(
            np.uint8
        ) * 255
        coverage_pct = (np.sum(mask_array > 0) / (height * width)) * 100
        self.model.train(was_training)
        return mask_array, cam, coverage_pct
