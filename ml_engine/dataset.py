"""Dataset ingestion and image transformation pipelines for Histopathology PNG patches."""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ml_engine.config import DATASET_DIR, IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD


def preprocess_image(image_input, target_size=IMAGE_SIZE):
    """Loads and standardizes an image to normalized PyTorch tensor (1, 3, H, W).
    
    Accepts:
        - file path (str or Path)
        - PIL Image
        - bytes buffer
    Returns:
        - (orig_rgb_np_normalized, tensor_ready_for_model)
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        import io
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Resize to model spatial dimensions (224, 224)
    pil_resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    img_np = np.array(pil_resized, dtype=np.float32)

    # Robust min-max normalization
    i_min, i_max = img_np.min(), img_np.max()
    if i_max > i_min:
        norm_img = (img_np - i_min) / (i_max - i_min)
    else:
        norm_img = np.zeros_like(img_np)

    # ImageNet standard normalization: (img - mean) / std
    mean = np.array(IMAGE_MEAN, dtype=np.float32)
    std = np.array(IMAGE_STD, dtype=np.float32)
    standardized = (norm_img - mean) / std

    # Channels-first tensor (1, 3, H, W)
    tensor = torch.tensor(
        np.ascontiguousarray(standardized).transpose(2, 0, 1),
        dtype=torch.float32
    ).unsqueeze(0)

    return norm_img, tensor


class HistologyDataset(Dataset):
    """PyTorch Dataset for Histopathology Patches."""
    def __init__(self, data_list, is_train=False):
        self.data_list = data_list
        self.is_train = is_train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        _, tensor = preprocess_image(item["image"])
        return {
            "image": tensor.squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "path": str(item["image"])
        }


def scan_dataset_split(split_name="Training"):
    """Scans subdirectories for a given split ('Training', 'Validation', 'Testing').
    
    Handles folder casing variations (e.g. 'Metastatic_Training' vs 'primary_validation').
    Returns a list of dicts: [{"image": Path, "label": int, "class_name": str}]
    """
    if not DATASET_DIR.exists():
        return []

    # Find the split directory matching case-insensitively
    split_dir = None
    for folder in DATASET_DIR.iterdir():
        if folder.is_dir() and folder.name.lower() == split_name.lower():
            split_dir = folder
            break

    if not split_dir:
        return []

    data = []
    for subfolder in split_dir.iterdir():
        if not subfolder.is_dir():
            continue
        sub_lower = subfolder.name.lower()
        if "primary" in sub_lower:
            label = 0
            cname = "Primary"
        elif "metastatic" in sub_lower:
            label = 1
            cname = "Metastatic"
        else:
            continue

        for f in subfolder.iterdir():
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                data.append({"image": f, "label": label, "class_name": cname})

    return data


def get_test_samples(count=5):
    """Retrieves sample images from Testing/ for quick frontend evaluation."""
    test_data = scan_dataset_split("Testing")
    primaries = [d for d in test_data if d["label"] == 0][:count]
    metastatics = [d for d in test_data if d["label"] == 1][:count]
    return primaries, metastatics

