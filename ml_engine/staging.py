"""Clinical Risk Staging and Decision Support Engine."""

from typing import Dict, Any
from ml_engine.config import STAGING_RULES, CLASS_NAMES


def get_clinical_stage(pred_idx: int, primary_prob: float, meta_prob: float) -> Dict[str, Any]:
    """Computes Stage 0 to Stage 3 stratification based on model prediction and confidence.
    
    Args:
        pred_idx: 0 for Primary, 1 for Metastatic.
        primary_prob: Probability percentage for Primary (0.0 to 100.0).
        meta_prob: Probability percentage for Metastatic (0.0 to 100.0).
        
    Returns:
        Dictionary containing stage_code, tier, risk, action, badge_color, and summary text.
    """
    if pred_idx == 0:  # Primary
        confidence = primary_prob
        rules = STAGING_RULES[0]
        if primary_prob >= rules["high_confidence"]["threshold"]:
            info = rules["high_confidence"]
        else:
            info = rules["low_confidence"]
    else:  # Metastatic
        confidence = meta_prob
        rules = STAGING_RULES[1]
        if meta_prob < rules["low_confidence"]["threshold"]:
            info = rules["low_confidence"]
        else:
            info = rules["high_confidence"]

    return {
        "pred_class": CLASS_NAMES[pred_idx],
        "pred_idx": pred_idx,
        "confidence": round(confidence, 2),
        "primary_prob": round(primary_prob, 2),
        "metastatic_prob": round(meta_prob, 2),
        "stage": info["stage"],
        "tier": info["tier"],
        "risk": info["risk"],
        "action": info["action"],
        "color": info["color"]
    }

