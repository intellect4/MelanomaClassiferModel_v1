"""Inference Service Singleton.

Loads model weights, manages Test-Time Augmentation (TTA),
generates Grad-CAM visual heatmaps, and computes clinical risk scores.
"""

import io
import time
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import torch
import torch.nn.functional as F

from ml_engine.config import DEVICE, BEST_MODEL_PATH, CLASS_NAMES
from ml_engine.model import MultiScaleAttentionMelanomaNet
from ml_engine.dataset import preprocess_image
from ml_engine.staging import get_clinical_stage
from ml_engine.explainability import GradCAMExplainer, generate_visualization_overlays


class MelanomaInferenceService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MelanomaInferenceService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = DEVICE
        self.model = MultiScaleAttentionMelanomaNet(num_classes=2).to(self.device)
        self.checkpoint_loaded = False

        if BEST_MODEL_PATH.exists():
            try:
                state = torch.load(BEST_MODEL_PATH, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state_dict = state["state_dict"]
                elif isinstance(state, dict) and "model" in state:
                    state_dict = state["model"]
                else:
                    state_dict = state

                model_dict = self.model.state_dict()
                matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(matched)
                self.model.load_state_dict(model_dict, strict=False)
                self.checkpoint_loaded = True
                print(f"Loaded trained checkpoint: {BEST_MODEL_PATH} ({len(matched)} matched layers)")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
        else:
            print("Notice: No trained checkpoint found in checkpoints/. Running in Architecture Preview mode.")

        self.model.eval()
        self.explainer = GradCAMExplainer(self.model)

    def reload_checkpoint(self) -> bool:
        """Reloads checkpoint if user drops best_melanoma_model.pth into checkpoints/."""
        self._initialize()
        return self.checkpoint_loaded

    def evaluate_with_tta(self, tensor_x: torch.Tensor) -> torch.Tensor:
        """4-View Test-Time Augmentation (TTA).
        
        Ensembles:
        1. Original view
        2. Horizontal flip
        3. Vertical flip
        4. 90-degree spatial rotation
        """
        tensor_x = tensor_x.to(self.device)
        with torch.no_grad():
            p1 = F.softmax(self.model(tensor_x), dim=1)
            p2 = F.softmax(self.model(torch.flip(tensor_x, dims=[3])), dim=1)
            p3 = F.softmax(self.model(torch.flip(tensor_x, dims=[2])), dim=1)
            p4 = F.softmax(self.model(torch.rot90(tensor_x, k=1, dims=[2, 3])), dim=1)
            avg_probs = (p1 + p2 + p3 + p4) / 4.0
        return avg_probs

    def diagnose_image(
        self,
        image_input,
        specimen_id: str = "Unknown",
        alpha: float = 0.45,
        salient_threshold: float = 0.60,
        return_base64: bool = True,
        use_tta: bool = True
    ) -> Dict[str, Any]:
        """Runs end-to-end diagnostic inference, Grad-CAM, and clinical staging."""
        start_time = time.time()
        
        # 1. Preprocess image
        orig_np, input_tensor = preprocess_image(image_input)
        input_tensor = input_tensor.to(self.device)

        # 2. Run TTA or Single-View Prediction
        if use_tta:
            probs = self.evaluate_with_tta(input_tensor)
        else:
            with torch.no_grad():
                probs = F.softmax(self.model(input_tensor), dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
        primary_prob = float(probs[0, 0].item() * 100.0)
        meta_prob = float(probs[0, 1].item() * 100.0)
        confidence = meta_prob if pred_idx == 1 else primary_prob

        # 3. Clinical Staging
        staging_info = get_clinical_stage(pred_idx, primary_prob, meta_prob)

        # 4. Generate Grad-CAM Heatmap
        cam = self.explainer.generate_heatmap(input_tensor, class_idx=pred_idx)

        # 5. Generate Visual Overlays
        pil_orig, pil_heat, pil_foci = generate_visualization_overlays(
            orig_np, cam, alpha=alpha, salient_threshold=salient_threshold
        )

        inference_ms = round((time.time() - start_time) * 1000.0, 2)

        result = {
            "specimen_id": specimen_id,
            "predicted_class": CLASS_NAMES[pred_idx],
            "predicted_idx": pred_idx,
            "confidence_percent": round(confidence, 2),
            "primary_probability": round(primary_prob, 2),
            "metastatic_probability": round(meta_prob, 2),
            "staging": staging_info,
            "tta_enhanced": use_tta,
            "has_trained_weights": self.checkpoint_loaded,
            "inference_time_ms": inference_ms,
            "pil_images": {
                "original": pil_orig,
                "heatmap": pil_heat,
                "salient_foci": pil_foci
            }
        }

        if return_base64:
            result["orig_image_base64"] = self._pil_to_base64(pil_orig)
            result["heatmap_overlay_base64"] = self._pil_to_base64(pil_heat)
            result["salient_foci_base64"] = self._pil_to_base64(pil_foci)

        return result

    @staticmethod
    def _pil_to_base64(pil_img: Image.Image) -> str:
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

