"""Explainable AI (XAI) Engine: Grad-CAM and Histopathological Tissue Masking.

Generates localized visual explanations highlighting cellular atypia and nuclear
clusters responsible for the model's Primary vs. Metastatic classification.
"""

import os
import tempfile
# Suppress matplotlib tempdir warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "melanoma_mpl"))

from typing import Tuple, Dict
import numpy as np
from PIL import Image, ImageFilter
import matplotlib
import matplotlib.cm as cm
import torch
import torch.nn.functional as F


class GradCAMExplainer:
    """Computes Gradient-weighted Class Activation Mapping (Grad-CAM) for DenseNet-121."""
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.model.eval()
        
        # Default to the final conv layer of the final dense block: denseblock4.denselayer16.conv2
        if target_layer is None:
            self.target_layer = self.model.features.denseblock4.denselayer16.conv2
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Generates raw 2D normalized Grad-CAM heatmap array in range [0, 1]."""
        self.model.zero_grad()
        
        # Ensure gradient tracking
        tensor_with_grad = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(tensor_with_grad)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        score.backward()

        # Global average pooling of gradients: weights across feature maps
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Only features with positive influence

        # Upsample to input tensor resolution (H, W)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()

        # Min-max scaling
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def apply_tissue_background_mask(heatmap: np.ndarray, orig_img_np: np.ndarray) -> np.ndarray:
    """Masks out non-tissue blank glass slide regions from the heatmap."""
    # Convert RGB [0, 1] to grayscale luminance
    gray = 0.299 * orig_img_np[:, :, 0] + 0.587 * orig_img_np[:, :, 1] + 0.114 * orig_img_np[:, :, 2]
    
    # Threshold dark/blank background
    mask = np.where(gray > (15.0 / 255.0), 1.0, 0.0).astype(np.float32)
    
    # Smooth mask boundaries using Pillow gaussian blur
    pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
    blurred_mask = np.array(pil_mask.filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32) / 255.0

    # Clean heatmap: zero out areas without tissue
    clean_cam = np.clip(heatmap * blurred_mask, 0.0, 1.0)
    return clean_cam


def generate_visualization_overlays(
    orig_img_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    salient_threshold: float = 0.60
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Produces 3 PIL visualization images:
    
    1. original_image: Clean input ROI patch
    2. blended_heatmap: Jet colormap blended on original tissue
    3. salient_foci: Highlighting top cellular foci (> threshold attention)
    """
    clean_cam = apply_tissue_background_mask(heatmap, orig_img_np)
    
    # 1. Original PIL Image
    orig_uint8 = np.clip(orig_img_np * 255.0, 0, 255).astype(np.uint8)
    pil_orig = Image.fromarray(orig_uint8)

    # 2. Blended Jet Heatmap
    cmap = matplotlib.colormaps["jet"]
    heatmap_rgba = cmap(clean_cam)
    heatmap_rgb = heatmap_rgba[:, :, :3]  # [0, 1]
    
    blended_np = (1.0 - alpha) * orig_img_np + alpha * heatmap_rgb
    blended_uint8 = np.clip(blended_np * 255.0, 0, 255).astype(np.uint8)
    pil_blended = Image.fromarray(blended_uint8)

    # 3. Salient Foci View (> 60% attention highlighted, low-attention dimmed)
    focus_view = orig_img_np.copy()
    salient_mask = np.where(clean_cam >= salient_threshold, 1.0, 0.35)
    for c in range(3):
        focus_view[:, :, c] = focus_view[:, :, c] * salient_mask
        
    focus_uint8 = np.clip(focus_view * 255.0, 0, 255).astype(np.uint8)
    pil_focus = Image.fromarray(focus_uint8)

    return pil_orig, pil_blended, pil_focus

