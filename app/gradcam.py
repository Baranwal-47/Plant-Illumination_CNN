"""Grad-CAM heatmap generation for CNN explainability."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _normalize_heatmap(cam):
    cam = cam.detach().cpu().numpy()
    high = np.percentile(cam, 99)
    low = np.percentile(cam, 5)
    if high <= low:
        return np.zeros_like(cam, dtype=np.float32)
    return np.clip((cam - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _heatmap_to_rgb(heatmap):
    """Create a high-contrast blue-to-red heatmap without plotting dependencies."""
    heatmap = np.clip(heatmap, 0.0, 1.0)
    red = (255 * np.clip(1.8 * heatmap - 0.35, 0.0, 1.0)).astype(np.uint8)
    green = (255 * np.clip(1.7 - np.abs(heatmap - 0.55) * 2.6, 0.0, 1.0)).astype(np.uint8)
    blue = (255 * np.clip(1.2 - 1.7 * heatmap, 0.0, 1.0)).astype(np.uint8)

    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return rgb


def create_gradcam_overlay(image, model, transform, device, class_index, alpha=0.72):
    """Return a high-contrast PIL Grad-CAM overlay for the selected class."""
    activations = []
    gradients = []
    target_layer = model.conv_block4

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        output = model(input_tensor)
        score = output[0, class_index]
        score.backward()

        if not activations or not gradients:
            raise RuntimeError("Could not capture model activations for Grad-CAM.")

        activation = activations[0]
        gradient = gradients[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activation, dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        heatmap_np = _normalize_heatmap(cam)
        heatmap_rgb = _heatmap_to_rgb(heatmap_np)

        original = image.convert("RGB")
        overlay = Image.fromarray(heatmap_rgb).resize(original.size)
        blended = Image.blend(original, overlay, alpha)
        return blended
    finally:
        forward_handle.remove()
        backward_handle.remove()
