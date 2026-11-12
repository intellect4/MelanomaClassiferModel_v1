# 🔬 Melanoma Histopathology AI Decision Support System

An end-to-end clinical AI application for **Primary vs. Metastatic Melanoma Subtyping** using a **MultiScale Attention DenseNet-121** network with **Convolutional Block Attention (CBAM)**, **4-View Test-Time Augmentation (TTA)**, **Grad-CAM Explainability**, and **Stage 0–3 Risk Stratification**.

---

## 🏗️ Architecture Overview

```
MelanomaClassifier/
├── DataSet/                            # Local 7,677 histology patches (1024x1024 PNG)
│   ├── Training/ [Primary_Training, Metastatic_Training]
│   ├── Validation/ [primary_validation, metastatic_validation]
│   └── Testing/ [primary_testing, metastatic_testing]
│
├── ml_engine/                          # Core Machine Learning Package
│   ├── config.py                       # Paths, device, hyperparameters, staging rules
│   ├── model.py                        # DenseNet-121 + CBAM Attention + Dual Pooling
│   ├── dataset.py                      # Dataset scanner & ImageNet standardization
│   ├── explainability.py               # Grad-CAM, tissue masking, salient foci
│   └── staging.py                      # Stage 0 to Stage 3 Clinical Decision Engine
│
├── backend/                            # FastAPI Production REST Service
│   ├── main.py                         # FastAPI server with CORS & endpoints
│   ├── schemas.py                      # Pydantic data models
│   └── service.py                      # Model inference singleton
│
├── frontend/                           # Responsive Clinical Dashboard
│   └── app.py                          # Streamlit application with 4-panel visualizer
│
├── checkpoints/                        # Model weights directory
│   └── best_melanoma_model.pth         # Place trained weights here
│
├── colab/                              # Fast GPU Cloud Training
│   ├── train_on_colab.ipynb            # Ready-to-run 1-click Colab notebook
│   └── train_on_colab.py               # Pure Python training script
│
├── run_frontend.bat                    # 1-Click launcher for Web Dashboard
├── run_backend.bat                     # 1-Click launcher for REST API
└── Melanoma_Classifier_Architecture_Plan.pdf # Full 6-Page Architecture Blueprint
```

---

## 🚀 Quick Start Guide

### 1. Launch the Clinical Frontend on your PC
Double-click `run_frontend.bat` (or run in terminal):
```powershell
python -m streamlit run frontend/app.py
```
Open your browser at **`http://localhost:8501`**.

The Clinical Workstation provides **4 Focused Dashboard Tabs**:
* **🔬 Diagnostic Assessment**: Upload custom histopathology slides directly with a 4-panel visualizer (Input ROI, Grad-CAM Heatmap overlay, Salient Nuclear Centers, Stage 0–3 Risk Badge & probability chart), an optional click-button **5th parameter** (Confusion Matrix & ROC Curve), and **Download Clinical Report (PDF)**.
* **🗂️ Search Analysis History**: Search and inspect previously analyzed specimens persisted in the local backend SQLite database (`backend/history_storage.db`) with filters and PDF re-download.
* **⚡ Batch Slide Triage**: Upload multiple biopsies simultaneously for automated priority triage sorting and CSV export.
* **🧬 Clinical Staging Guidelines**: AJCC-aligned Stage 0–3 clinical management matrix and Grad-CAM formulation.

---

### 2. Launch the REST API Backend (Optional)
Double-click `run_backend.bat` (or run in terminal):
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API documentation will be available at: **`http://127.0.0.1:8000/docs`**.

---

### 3. Training on Google Colab (One-Time, ~25 Mins)
Because your PC has an Intel Core i5-13500H CPU, training 7,338 images from scratch is 50x faster on Google Colab's free **NVIDIA Tesla T4 GPU**.

1. Upload `colab/train_on_colab.ipynb` to [Google Colab](https://colab.research.google.com).
2. Set Runtime: **Runtime &rarr; Change runtime type &rarr; T4 GPU**.
3. Click **Run All**.
4. Once completed (~25 minutes), Colab will automatically download `best_melanoma_model.pth`.
5. Move `best_melanoma_model.pth` into your local `checkpoints/` folder.
6. Refresh your frontend &mdash; the model will immediately use the newly trained parameters!

