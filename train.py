"""
train.py
YOLOv8 Instance Segmentation -- MLflow experiment tracking
Dataset: COCO128-seg (toy dataset, ~128 images, auto-downloads)

Usage:
    python train.py                          # default hyperparams
    python train.py --epochs 10 --lr0 0.01   # custom run
"""

import argparse
import json
import os
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
import yaml
from ultralytics import YOLO

# Disable ultralytics built-in MLflow plugin to avoid conflicts
os.environ["MLFLOW_TRACKING_URI"] = ""
try:
    from ultralytics.utils import SETTINGS
    SETTINGS.update({"mlflow": False})
except Exception:
    pass

# Constants
EXPERIMENT_NAME = "YOLOv8-Segmentation"
DATASET         = "coco128-seg.yaml"
BASE_MODEL      = "yolov8n-seg.pt"
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


# MLflow model wrapper
class YOLOSegWrapper(mlflow.pyfunc.PythonModel):
    """Wraps a saved YOLOv8 segmentation model for MLflow serving."""

    def load_context(self, context):
        from ultralytics import YOLO
        self.model = YOLO(context.artifacts["model_path"])

    def predict(self, context, model_input):
        results_out = []
        for _, row in model_input.iterrows():
            res = self.model(row["image_path"])
            r   = res[0]
            entry = {
                "image":       row["image_path"],
                "num_objects": int(len(r.boxes)),
                "classes":     [r.names[int(c)] for c in r.boxes.cls.tolist()],
                "confidences": [round(float(c), 4) for c in r.boxes.conf.tolist()],
                "boxes_xyxy":  r.boxes.xyxy.tolist(),
            }
            results_out.append(entry)
        return results_out


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8-seg and log to MLflow")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--imgsz",        type=int,   default=640)
    p.add_argument("--batch",        type=int,   default=8)
    p.add_argument("--lr0",          type=float, default=0.01)
    p.add_argument("--lrf",          type=float, default=0.01)
    p.add_argument("--momentum",     type=float, default=0.937)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--augment",      action="store_true")
    p.add_argument("--model",        type=str,   default=BASE_MODEL)
    p.add_argument("--dataset",      type=str,   default=DATASET)
    p.add_argument("--run_name",     type=str,   default=None)
    return p.parse_args()


def find_results_dir(run_name: str, save_dir_hint: Path) -> Path:
    """
    Robustly find where ultralytics actually saved the results.
    Searches multiple candidate paths.
    """
    candidates = [
        save_dir_hint,
        save_dir_hint.resolve(),
        Path("runs/segment") / run_name,
        Path("runs/segment/runs/segment") / run_name,
    ]
    for c in candidates:
        weights = c / "weights"
        if weights.exists():
            print(f"[info] Found results at: {c}")
            return c

    # Deep search as last resort
    print("[info] Searching for results directory...")
    for p in sorted(Path(".").rglob("*/weights/best.pt")):
        candidate = p.parent.parent
        if run_name in str(candidate):
            print(f"[info] Found via search: {candidate}")
            return candidate

    print(f"[warn] Could not find results dir, using hint: {save_dir_hint}")
    return save_dir_hint


def collect_yolo_metrics(results_dir: Path) -> dict:
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        return {}
    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    if not rows:
        return {}
    last = rows[-1]
    metrics = {}
    key_map = {
        "metrics/precision(B)":  "precision_box",
        "metrics/recall(B)":     "recall_box",
        "metrics/mAP50(B)":      "mAP50_box",
        "metrics/mAP50-95(B)":   "mAP50_95_box",
        "metrics/precision(M)":  "precision_mask",
        "metrics/recall(M)":     "recall_mask",
        "metrics/mAP50(M)":      "mAP50_mask",
        "metrics/mAP50-95(M)":   "mAP50_95_mask",
        "train/box_loss":        "train_box_loss",
        "train/seg_loss":        "train_seg_loss",
        "train/cls_loss":        "train_cls_loss",
        "val/box_loss":          "val_box_loss",
        "val/seg_loss":          "val_seg_loss",
        "val/cls_loss":          "val_cls_loss",
    }
    for src, dst in key_map.items():
        if src in last:
            try:
                metrics[dst] = float(last[src])
            except ValueError:
                pass
    return metrics


def log_per_epoch_metrics(results_dir: Path):
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        return
    import csv
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
    metric_cols = {
        "metrics/mAP50(M)":     "mAP50_mask",
        "metrics/mAP50(B)":     "mAP50_box",
        "train/box_loss":       "train_box_loss",
        "train/seg_loss":       "train_seg_loss",
        "val/box_loss":         "val_box_loss",
        "val/seg_loss":         "val_seg_loss",
        "metrics/precision(M)": "precision_mask",
        "metrics/recall(M)":    "recall_mask",
    }
    for epoch, row in enumerate(rows):
        for src, dst in metric_cols.items():
            if src in row:
                try:
                    mlflow.log_metric(dst, float(row[src]), step=epoch)
                except Exception:
                    pass


def log_image_artifacts(results_dir: Path):
    for png in results_dir.glob("*.png"):
        mlflow.log_artifact(str(png), artifact_path="plots")
    for subdir in results_dir.iterdir():
        if subdir.is_dir() and subdir.name != "weights":
            for png in subdir.glob("*.png"):
                mlflow.log_artifact(str(png), artifact_path=f"plots/{subdir.name}")


def log_dataset_info(dataset_yaml: str):
    try:
        from ultralytics.utils import DATASETS_DIR
        yaml_path = DATASETS_DIR / dataset_yaml
        if yaml_path.exists():
            with open(yaml_path) as f:
                info = yaml.safe_load(f)
            mlflow.log_dict(info, "dataset/dataset_info.json")
            mlflow.set_tag("dataset.nc",   str(info.get("nc", "?")))
            mlflow.set_tag("dataset.name", info.get("dataset_name", dataset_yaml))
            val_path = Path(info.get("path", "")) / "images" / "val"
            if val_path.exists():
                n_val = len(list(val_path.glob("*.jpg")) + list(val_path.glob("*.png")))
                mlflow.log_metric("dataset_val_images", n_val)
    except Exception as e:
        print(f"[dataset info] {e}")


def log_val_images(results_dir: Path, n: int = 10):
    val_imgs = list(results_dir.glob("val_batch*.jpg"))
    for img in val_imgs[:n]:
        mlflow.log_artifact(str(img), artifact_path="val_predictions")


def main():
    args = parse_args()

    # Set MLflow URI after disabling ultralytics plugin
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = args.run_name or f"yolov8n-seg_ep{args.epochs}_lr{args.lr0}"

    print(f"\n{'='*60}")
    print(f"  YOLOv8 Segmentation Training")
    print(f"  Experiment : {EXPERIMENT_NAME}")
    print(f"  Run        : {run_name}")
    print(f"  Tracking   : {MLFLOW_URI}")
    print(f"{'='*60}\n")

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.set_tags({
            "framework":    "ultralytics",
            "model_type":   "YOLOv8-seg",
            "task":         "instance_segmentation",
            "dataset":      args.dataset,
            "base_weights": args.model,
        })

        mlflow.log_params({
            "epochs":        args.epochs,
            "imgsz":         args.imgsz,
            "batch":         args.batch,
            "lr0":           args.lr0,
            "lrf":           args.lrf,
            "momentum":      args.momentum,
            "weight_decay":  args.weight_decay,
            "augment":       args.augment,
            "base_model":    args.model,
            "dataset":       args.dataset,
            "optimizer":     "SGD",
        })

        log_dataset_info(args.dataset)

        model = YOLO(args.model)
        t0    = time.time()

        train_results = model.train(
            data         = args.dataset,
            epochs       = args.epochs,
            imgsz        = args.imgsz,
            batch        = args.batch,
            lr0          = args.lr0,
            lrf          = args.lrf,
            momentum     = args.momentum,
            weight_decay = args.weight_decay,
            name         = run_name,
            exist_ok     = True,
            plots        = True,
            save         = True,
            verbose      = True,
        )
        train_time = time.time() - t0
        mlflow.log_metric("train_time_seconds", round(train_time, 2))

        # Robustly find the actual results directory
        hint = Path(str(train_results.save_dir))
        results_dir = find_results_dir(run_name, hint)
        print(f"[info] Using results dir: {results_dir}")

        log_per_epoch_metrics(results_dir)

        final_metrics = collect_yolo_metrics(results_dir)
        if final_metrics:
            mlflow.log_metrics(final_metrics)
            print(f"\n[metrics] {json.dumps(final_metrics, indent=2)}")

        log_image_artifacts(results_dir)
        log_val_images(results_dir)

        csv_path = results_dir / "results.csv"
        if csv_path.exists():
            mlflow.log_artifact(str(csv_path), artifact_path="metrics")

        args_yaml = results_dir / "args.yaml"
        if args_yaml.exists():
            mlflow.log_artifact(str(args_yaml), artifact_path="config")

        best_pt = results_dir / "weights" / "best.pt"
        last_pt = results_dir / "weights" / "last.pt"

        print(f"[info] best.pt exists: {best_pt.exists()} -> {best_pt}")
        print(f"[info] last.pt exists: {last_pt.exists()} -> {last_pt}")

        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
        if last_pt.exists():
            mlflow.log_artifact(str(last_pt), artifact_path="weights")

        model_pt = best_pt if best_pt.exists() else (last_pt if last_pt.exists() else None)

        if model_pt is None:
            print("[warn] No .pt weights found, skipping pyfunc model logging")
        else:
            print(f"[info] Logging model from: {model_pt}")
            artifacts = {"model_path": str(model_pt.resolve())}
            conda_env = {
                "name": "yolov8-seg",
                "channels": ["defaults"],
                "dependencies": [
                    "python=3.10",
                    "pip",
                    {"pip": ["ultralytics>=8.0", "mlflow>=2.0", "numpy", "pandas"]},
                ],
            }
            model_info = mlflow.pyfunc.log_model(
                artifact_path         = "model",
                python_model          = YOLOSegWrapper(),
                artifacts             = artifacts,
                conda_env             = conda_env,
                registered_model_name = "YOLOv8-Seg",
            )
            print(f"\n[model] logged: {model_info.model_uri}")

        # Final validation pass
        print("\n[val] Running final validation pass...")
        try:
            val_res = model.val(data=args.dataset, imgsz=args.imgsz, split="val")
            mlflow.log_metrics({
                "val_mAP50_box_final":     float(val_res.box.map50),
                "val_mAP50_95_box_final":  float(val_res.box.map),
                "val_mAP50_mask_final":    float(val_res.seg.map50),
                "val_mAP50_95_mask_final": float(val_res.seg.map),
            })
        except Exception as e:
            print(f"[val] {e}")

        print(f"\n{'='*60}")
        print(f"  Run ID : {run.info.run_id}")
        print(f"  URI    : {MLFLOW_URI}/#/experiments")
        print(f"{'='*60}\n")

    return run.info.run_id


if __name__ == "__main__":
    main()
