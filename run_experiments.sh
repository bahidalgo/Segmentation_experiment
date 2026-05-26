#!/usr/bin/env bash


set -euo pipefail

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-https://demo.mlflow.org}"

echo ""
echo "============================================================"
echo "  YOLOv8-Seg — Hyperparameter Sweep"
echo "  MLflow URI: $MLFLOW_TRACKING_URI"
echo "============================================================"
echo ""


echo "[run 1/4] Baseline — lr0=0.01, epochs=5"
python train.py \
  --epochs 5 \
  --lr0 0.01 \
  --lrf 0.01 \
  --batch 8 \
  --imgsz 640 \
  --run_name "baseline_lr0.01_ep5"


echo "[run 2/4] Higher LR — lr0=0.05, epochs=5"
python train.py \
  --epochs 5 \
  --lr0 0.05 \
  --lrf 0.01 \
  --batch 8 \
  --imgsz 640 \
  --run_name "high_lr_lr0.05_ep5"


echo "[run 3/4] More epochs — lr0=0.01, epochs=10"
python train.py \
  --epochs 10 \
  --lr0 0.01 \
  --lrf 0.005 \
  --batch 8 \
  --imgsz 640 \
  --run_name "longer_lr0.01_ep10"


echo "[run 4/4] Small imgsz — imgsz=320, epochs=5"
python train.py \
  --epochs 5 \
  --lr0 0.01 \
  --lrf 0.01 \
  --batch 16 \
  --imgsz 320 \
  --run_name "small_imgsz320_ep5"

echo ""
echo "============================================================"
echo "  All experiments done!"
echo "  View results at: $MLFLOW_TRACKING_URI"
echo "============================================================"
echo ""
