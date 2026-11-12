"""Pydantic data schemas for API requests and diagnostic responses."""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class StagingDetail(BaseModel):
    stage: str
    tier: str
    risk: str
    action: str
    color: str


class PredictionResult(BaseModel):
    specimen_id: str
    predicted_class: str
    predicted_idx: int
    confidence_percent: float
    primary_probability: float
    metastatic_probability: float
    staging: StagingDetail
    tta_enhanced: bool = True
    has_trained_weights: bool
    inference_time_ms: float
    
    # Base64 encoded PNG visual representations
    orig_image_base64: Optional[str] = None
    heatmap_overlay_base64: Optional[str] = None
    salient_foci_base64: Optional[str] = None


class SampleImageInfo(BaseModel):
    sample_id: str
    filename: str
    class_label: str
    split: str
    full_path: str


class HealthStatus(BaseModel):
    status: str
    device: str
    model_architecture: str
    checkpoint_loaded: bool
    checkpoint_path: str
    num_classes: int
    class_names: List[str]

