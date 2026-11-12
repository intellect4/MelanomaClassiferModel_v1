"""Configuration settings for Melanoma Classifier ML Engine."""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "DataSet"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_melanoma_model.pth"

# Training / Processing Parameters
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2
CLASS_NAMES = ["Primary", "Metastatic"]
CLASS_TO_IDX = {"Primary": 0, "Metastatic": 1}

# Normalization (ImageNet standards)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Hardware Device Detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clinical Staging Definitions
STAGING_RULES = {
    0: {  # Primary Melanoma
        "high_confidence": {
            "threshold": 85.0,
            "stage": "Stage 0",
            "tier": "In Situ / Low Cellular Atypia",
            "risk": "Low Risk",
            "action": "Routine surveillance; no systemic intervention indicated.",
            "color": "#2ca02c"  # Green
        },
        "low_confidence": {
            "stage": "Stage 1",
            "tier": "Early Localized Invasive Primary",
            "risk": "Moderate Risk",
            "action": "Diagnostic wide local excision with clear histological margins.",
            "color": "#1f77b4"  # Blue
        }
    },
    1: {  # Metastatic Melanoma
        "low_confidence": {
            "threshold": 85.0,
            "stage": "Stage 2",
            "tier": "Regional / Micro-Metastatic Spread",
            "risk": "High Risk",
            "action": "Sentinel Lymph Node Biopsy (SLNB) & immunohistochemical staging.",
            "color": "#ff7f0e"  # Orange
        },
        "high_confidence": {
            "stage": "Stage 3",
            "tier": "Macroscopic / Frank Metastatic Infiltration",
            "risk": "Critical / Urgent",
            "action": "Immediate oncology referral; staging PET-CT & targeted immunotherapy.",
            "color": "#d62728"  # Red
        }
    }
}

