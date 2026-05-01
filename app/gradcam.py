"""Grad-CAM heatmap generation for CNN explainability."""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class GradCamResult:
    heatmap: Image.Image
    overlay: Image.Image


def _resize_heatmap(heatmap, size):
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_image = heatmap_image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(heatmap_image).astype(np.float32) / 255.0


def _apply_jet_colormap(heatmap):
    """Approximate OpenCV's JET colormap with NumPy."""
    heatmap = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * heatmap - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * heatmap - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * heatmap - 1.0), 0.0, 1.0)
    return np.uint8(255 * np.stack([red, green, blue], axis=-1))


def create_gradcam_visualization(image, model, transform, device, class_index):
    """Create class-specific Grad-CAM heatmap and overlay images.

    The heatmap is computed from the trained CNN by backpropagating the predicted
    class score into the first convolution of the model's final convolution block.
    """
    activations = None
    gradients = None
    target_layer = model.conv_block4[0]

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        original = image.convert("RGB")
        input_tensor = transform(original).unsqueeze(0).to(device)
        output = model(input_tensor)
        output[:, class_index].backward()

        if activations is None or gradients is None:
            raise RuntimeError("Could not capture activations or gradients for Grad-CAM.")

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        weighted_activations = activations.clone()
        for channel_idx in range(weighted_activations.size(1)):
            weighted_activations[:, channel_idx, :, :] *= pooled_gradients[channel_idx]

        heatmap = torch.mean(weighted_activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        max_value = np.max(heatmap)
        if max_value > 0:
            heatmap = heatmap / max_value

        original_np = np.array(original)
        heatmap = _resize_heatmap(heatmap, original.size)
        heatmap_rgb = _apply_jet_colormap(heatmap)
        overlay = np.clip(
            (original_np.astype(np.float32) * 0.6)
            + (heatmap_rgb.astype(np.float32) * 0.4),
            0,
            255,
        ).astype(np.uint8)

        return GradCamResult(
            heatmap=Image.fromarray(heatmap_rgb),
            overlay=Image.fromarray(overlay),
        )
    finally:
        forward_handle.remove()
        backward_handle.remove()
