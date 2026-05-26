# YOLOv8 Instance Segmentation — MLflow Experiment Tracking

> **Tarea 2 – Plataformas de Big Data para Data Science**  
> Magíster en Inteligencia Artificial — Pontificia Universidad Católica de Chile

---

## Overview

This project modernizes an ML development workflow using:

| Tool | Purpose |
|------|---------|
| **Git** | Version control |
| **Python venv** | Reproducible environments (`requirements.txt`) |
| **YOLOv8-seg** | Instance segmentation model (Ultralytics) |
| **MLflow** | Experiment tracking, model registry, and serving |

**Dataset**: COCO128-seg — a 128-image subset of COCO with segmentation masks, downloaded automatically by Ultralytics on first run. No manual setup required.

**MLflow Tracking Server**: [`https://demo.mlflow.org`](https://demo.mlflow.org)

---

## Project Structure

```
segmentation_experiment/
├── train.py              # Main training script (YOLOv8 + MLflow)
├── predict.py            # Inference: online API + batch mode
├── run_experiments.sh    # Runs 4 experiments varying hyperparams
├── setup_env.sh          # Creates & configures virtual environment
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/bahidalgo/Segmentation_experiment.git
cd Segmentation_experiment
```

### 2. Set up the virtual environment

```bash
chmod +x setup_env.sh
./setup_env.sh
source .venv/bin/activate
```

### 3. Run a single training experiment

```bash
# Uses COCO128-seg, downloads automatically on first run
python train.py --epochs 5 --lr0 0.01
```

### 4. Run the full hyperparameter sweep (4 experiments)

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

### 5. View results in MLflow UI

Open [https://demo.mlflow.org](https://demo.mlflow.org) and navigate to the **YOLOv8-Segmentation** experiment.

---

## What Gets Logged to MLflow

### Parameters
| Parameter | Description |
|-----------|-------------|
| `epochs` | Number of training epochs |
| `imgsz` | Input image size |
| `batch` | Batch size |
| `lr0` | Initial learning rate |
| `lrf` | Final learning rate factor |
| `momentum` | SGD momentum |
| `weight_decay` | L2 regularization |
| `base_model` | Pre-trained weights used |
| `dataset` | Dataset YAML used |

### Metrics (per epoch + final)
| Metric | Description |
|--------|-------------|
| `mAP50_mask` | Mask mAP at IoU 0.50 |
| `mAP50_95_mask` | Mask mAP at IoU 0.50:0.95 |
| `mAP50_box` | Box mAP at IoU 0.50 |
| `precision_mask` / `recall_mask` | Segmentation P/R |
| `train_box_loss` / `train_seg_loss` | Training losses |
| `val_box_loss` / `val_seg_loss` | Validation losses |
| `train_time_seconds` | Total training duration |

### Artifacts
- `plots/` — Loss curves, PR curves, confusion matrix, F1 curve, label distribution
- `val_predictions/` — Validation batch images with predicted masks
- `weights/best.pt` — Best model weights
- `weights/last.pt` — Final epoch weights
- `metrics/results.csv` — Full per-epoch metrics CSV
- `config/args.yaml` — Full training configuration
- `dataset/dataset_info.json` — Dataset metadata
- `model/` — Logged MLflow `pyfunc` model (for serving)

---

## Model Serving

### Register and serve the model locally

After training, the model is automatically registered in the **MLflow Model Registry** as `YOLOv8-Seg`.

```bash
# Set tracking URI
export MLFLOW_TRACKING_URI=https://demo.mlflow.org

# Serve the model on port 5001
mlflow models serve -m "models:/YOLOv8-Seg/1" -p 5001 --no-conda
```

---

## Prediction

### Online (REST API)

While the model server is running:

```bash
# Download sample COCO images + run online prediction
python predict.py --mode online --download --images test_images/
```

Or with curl:

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {"image_path": "test_images/000000039769.jpg"}
    ]
  }'
```

### Batch (direct model load)

```bash
python predict.py --mode batch \
  --model_uri "models:/YOLOv8-Seg/1" \
  --download \
  --images test_images/
```

Results are saved to `batch_results.json`.

---

## Experiment Comparison

After running `run_experiments.sh`, you can compare all 4 runs in the MLflow UI:

| Run | epochs | lr0 | imgsz | Notes |
|-----|--------|-----|-------|-------|
| `baseline_lr0.01_ep5` | 5 | 0.010 | 640 | Baseline |
| `high_lr_lr0.05_ep5` | 5 | 0.050 | 640 | Higher LR |
| `longer_lr0.01_ep10` | 10 | 0.010 | 640 | More epochs |
| `small_imgsz320_ep5` | 5 | 0.010 | 320 | Smaller input |

---

## Git Workflow

```bash
git add .
git commit -m "feat: add training run results"
git push origin main
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `https://demo.mlflow.org` | MLflow server URL |

---

## Dependencies

See [`requirements.txt`](requirements.txt). Key packages:

- `ultralytics>=8.2.0` — YOLOv8
- `mlflow>=2.12.0` — Experiment tracking & serving
- `torch>=2.0.0` — Deep learning backend
- `pandas`, `numpy`, `opencv-python` — Data utilities
