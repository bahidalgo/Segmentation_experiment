"""
predict.py
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests



def collect_images(folder: str, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    p = Path(folder)
    if not p.exists():
        sys.exit(f"[error] folder not found: {folder}")
    imgs = [str(f) for f in p.iterdir() if f.suffix.lower() in extensions]
    if not imgs:
        sys.exit(f"[error] no images found in {folder}")
    print(f"[info] found {len(imgs)} image(s) in {folder}")
    return imgs


def pretty(results: list):
    for r in results:
        print(f"\n  image      : {r.get('image', '?')}")
        print(f"  objects    : {r.get('num_objects', 0)}")
        print(f"  classes    : {r.get('classes', [])}")
        print(f"  confidences: {r.get('confidences', [])}")



def predict_online(images: list, host: str = "127.0.0.1", port: int = 5001):
    """POST to the MLflow model REST API."""
    url = f"http://{host}:{port}/invocations"

    # Build dataframe-split payload
    records = [{"image_path": img} for img in images]
    payload = {
        "dataframe_records": records
    }

    print(f"\n[online] POST → {url}")
    print(f"[online] payload images: {images}")
    t0  = time.time()
    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )
        elapsed = time.time() - t0
        resp.raise_for_status()
        results = resp.json()
        print(f"[online] ✅ response in {elapsed:.2f}s")
        if isinstance(results, dict) and "predictions" in results:
            pretty(results["predictions"])
        else:
            pretty(results if isinstance(results, list) else [results])
        return results
    except requests.exceptions.ConnectionError:
        sys.exit(
            "[error] Cannot connect. Is `mlflow models serve` running?\n"
            "  Run: mlflow models serve -m models:/YOLOv8-Seg/1 -p 5001 --no-conda"
        )
    except requests.exceptions.HTTPError as e:
        sys.exit(f"[error] HTTP {resp.status_code}: {resp.text}")



def predict_batch(images: list, model_uri: str):
    """Load model directly from MLflow and run batch inference."""
    import mlflow
    import pandas as pd

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://demo.mlflow.org")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"\n[batch] Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    df = pd.DataFrame({"image_path": images})
    print(f"[batch] Running inference on {len(images)} image(s)…")
    t0      = time.time()
    results = model.predict(df)
    elapsed = time.time() - t0

    print(f"[batch] ✅ done in {elapsed:.2f}s")
    pretty(results)

    # Save results to JSON
    out = Path("batch_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[batch] Results saved → {out}")
    return results



def download_demo_images(dest: str = "test_images"):
    """Download a handful of COCO val images for quick testing."""
    dest_p = Path(dest)
    dest_p.mkdir(exist_ok=True)

    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats
        "http://images.cocodataset.org/val2017/000000397133.jpg",  # person+sports
        "http://images.cocodataset.org/val2017/000000037777.jpg",  # kitchen
    ]
    for url in urls:
        fname = dest_p / url.split("/")[-1]
        if fname.exists():
            continue
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            fname.write_bytes(r.content)
            print(f"[download] {fname}")
        except Exception as e:
            print(f"[download] failed {url}: {e}")
    return dest



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",      choices=["online", "batch"], default="batch")
    p.add_argument("--images",    default="test_images",  help="Folder with test images")
    p.add_argument("--model_uri", default="models:/YOLOv8-Seg/1", help="MLflow model URI (batch mode)")
    p.add_argument("--host",      default="127.0.0.1", help="API host (online mode)")
    p.add_argument("--port",      type=int, default=5001, help="API port (online mode)")
    p.add_argument("--download",  action="store_true", help="Download sample COCO images first")
    return p.parse_args()


def main():
    args = parse_args()

    if args.download:
        download_demo_images(args.images)

    images = collect_images(args.images)

    if args.mode == "online":
        predict_online(images, host=args.host, port=args.port)
    else:
        predict_batch(images, model_uri=args.model_uri)


if __name__ == "__main__":
    main()
