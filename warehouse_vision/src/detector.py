"""Thin wrapper around the fine-tuned YOLOv8n barcode-region detector (locates barcodes only)."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def load_detector(weights: Path):
    """Returns (model|None, message). Never raises for a missing file / missing library."""
    weights = Path(weights)
    if not weights.is_file():
        return None, f"Detector weights not found at {weights}. Train with: python scripts/05_train_detector.py"
    try:
        from ultralytics import YOLO
        return YOLO(str(weights)), ""
    except Exception as e:
        return None, f"Could not load detector ({type(e).__name__}: {e})"


def detect(model, img_bgr: np.ndarray, conf: float = 0.25, imgsz: int = 640) -> list[dict]:
    r = model.predict(img_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for b in r.boxes:
        out.append({"box": [float(v) for v in b.xyxy[0].tolist()], "conf": float(b.conf[0])})
    return sorted(out, key=lambda d: -d["conf"])
