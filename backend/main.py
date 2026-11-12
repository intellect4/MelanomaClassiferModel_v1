"""FastAPI Server Entrypoint for Melanoma Histopathology AI Decision Support."""

import hashlib
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml_engine.config import DEVICE, BEST_MODEL_PATH, CLASS_NAMES, DATASET_DIR
from ml_engine.dataset import scan_dataset_split
from backend.schemas import HealthStatus, PredictionResult, SampleImageInfo
from backend.service import MelanomaInferenceService
from backend.history import save_analysis_record, search_history, delete_history_record

app = FastAPI(
    title="Melanoma Histopathology AI Decision Support API",
    description="REST API for Deep Learning Subtyping (Primary vs. Metastatic) with Grad-CAM and Staging Engine",
    version="1.0.0"
)

# Enable CORS for cross-origin web/app requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inference Singleton
service = MelanomaInferenceService()


@app.get("/")
def root():
    """Root landing endpoint for API health and documentation discovery."""
    return {
        "service": "Melanoma Histopathology AI Decision Support API",
        "status": "ONLINE",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/api/v1/predict",
        "history_url": "/api/v1/history",
    }


@app.get("/health", response_model=HealthStatus)
def health_check():
    """Returns system status, device, and model checkpoint state."""
    return HealthStatus(
        status="ONLINE",
        device=str(service.device),
        model_architecture="MultiScaleAttentionMelanomaNet (DenseNet121 + CBAM)",
        checkpoint_loaded=service.checkpoint_loaded,
        checkpoint_path=str(BEST_MODEL_PATH),
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES
    )


@app.get("/api/v1/samples", response_model=List[SampleImageInfo])
def list_test_samples(limit_per_class: int = Query(5, ge=1, le=50)):
    """Retrieves verified histopathology test slides from the local DataSet/Testing folder."""
    all_test = scan_dataset_split("Testing")
    primaries = [d for d in all_test if d["label"] == 0][:limit_per_class]
    metastatics = [d for d in all_test if d["label"] == 1][:limit_per_class]

    result = []
    for idx, item in enumerate(primaries + metastatics):
        result.append(SampleImageInfo(
            sample_id=f"test_{idx+1:03d}",
            filename=item["image"].name,
            class_label=item["class_name"],
            split="Testing",
            full_path=str(item["image"])
        ))
    return result


@app.post("/api/v1/predict", response_model=PredictionResult)
async def predict_histology_slide(
    file: Optional[UploadFile] = File(None),
    sample_path: Optional[str] = Form(None),
    alpha: float = Form(0.45),
    salient_threshold: float = Form(0.60)
):
    """Predicts Primary vs. Metastatic subtyping with Grad-CAM and Clinical Staging.
    
    Accepts either:
    - An uploaded file (.png, .jpg, .tif)
    - A local sample path from DataSet/Testing
    """
    if file is not None:
        file_bytes = await file.read()
        specimen_id = file.filename
        image_input = file_bytes
        image_hash = hashlib.sha256(file_bytes).hexdigest()
    elif sample_path is not None:
        p = Path(sample_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Sample file not found: {sample_path}")
        specimen_id = p.name
        image_input = str(p)
        image_hash = hashlib.sha256(p.read_bytes()).hexdigest()
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded file or sample_path.")

    try:
        diagnosis = service.diagnose_image(
            image_input=image_input,
            specimen_id=specimen_id,
            alpha=alpha,
            salient_threshold=salient_threshold,
            return_base64=True
        )
        save_analysis_record(diagnosis, image_hash=image_hash, alpha=alpha, salient_threshold=salient_threshold)
        return PredictionResult(**diagnosis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic analysis failed: {str(e)}")


@app.get("/api/v1/history")
def get_analysis_history(
    q: str = Query("", description="Search specimen ID, report ID, or risk tier"),
    subtype: str = Query("All", description="Filter by Primary, Metastatic, or All"),
    stage: str = Query("All", description="Filter by Stage 0..3 or All"),
    limit: int = Query(50, ge=1, le=200)
):
    """Queries analyzed image history stored in backend/history_storage.db."""
    return search_history(query=q, subtype_filter=subtype, stage_filter=stage, limit=limit)


@app.delete("/api/v1/history/{record_id}")
def remove_analysis_history(record_id: int):
    """Deletes a record from backend/history_storage.db."""
    deleted = delete_history_record(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History record not found")
    return {"deleted": True, "record_id": record_id}


@app.post("/api/v1/reload-checkpoint")
def reload_model_checkpoint():
    """Reloads model weights if user drops best_melanoma_model.pth into checkpoints/."""
    loaded = service.reload_checkpoint()
    return {
        "success": loaded,
        "checkpoint_loaded": loaded,
        "message": "Weights reloaded successfully." if loaded else "No checkpoint found at checkpoints/best_melanoma_model.pth"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

