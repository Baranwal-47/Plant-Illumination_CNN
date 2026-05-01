"""Grad-CAM heatmap generation for CNN explainability."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _heatmap_to_rgb(heatmap):
    """Create a simple red/yellow heatmap without extra plotting dependencies."""
    heatmap = np.clip(heatmap, 0.0, 1.0)
    red = np.full_like(heatmap, 255)
    green = (255 * np.power(heatmap, 0.7)).astype(np.uint8)
    blue = np.zeros_like(green)
    alpha = (255 * heatmap).astype(np.uint8)

    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return rgb, alpha


def create_gradcam_overlay(image, model, transform, device, class_index, alpha=0.45):
    """Return a PIL image with a Grad-CAM overlay for the selected class."""
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

        cam_min = cam.min()
        cam_max = cam.max()
        if torch.isclose(cam_max, cam_min):
            heatmap = torch.zeros_like(cam)
        else:
            heatmap = (cam - cam_min) / (cam_max - cam_min)

        heatmap_np = heatmap.detach().cpu().numpy()
        heatmap_rgb, heatmap_alpha = _heatmap_to_rgb(heatmap_np)

        original = image.convert("RGB")
        overlay = Image.fromarray(heatmap_rgb).resize(original.size)
        mask = Image.fromarray(heatmap_alpha).resize(original.size)
        blended = Image.blend(original, overlay, alpha)
        return Image.composite(blended, original, mask)
    finally:
        forward_handle.remove()
        backward_handle.remove()
