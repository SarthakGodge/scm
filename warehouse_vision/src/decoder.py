"""Barcode decoding with pyzbar (deterministic library - NOT a trained model).
Detection confidence (YOLO) and decoding are separate: a decode either returns a value or it does not."""
from __future__ import annotations
import numpy as np
import cv2

try:
    from pyzbar import pyzbar
    DECODER_AVAILABLE, DECODER_ERROR = True, ""
except Exception as e:          # missing libzbar0 or pyzbar
    pyzbar, DECODER_AVAILABLE, DECODER_ERROR = None, False, str(e)

PADS = (0.15, 0.4, 0.8)   # progressive retry; chosen on train+val GT crops only (91.5% vs 87.7% for a single 0.15 pad)
LINEAR_FIRST = {"EAN13": 0, "EAN8": 0, "UPCA": 0, "UPCE": 0, "ISBN13": 0, "ISBN10": 0, "CODE128": 1, "CODE39": 1, "I25": 1, "CODE93": 1}


def crop_with_padding(img_bgr: np.ndarray, box_xyxy, pad: float = 0.15) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    return img_bgr[y1:y2, x1:x2]


def _variants(img):
    """Cheap preprocessing variants tried in order until one decodes."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    yield "original", img
    yield "gray", g
    big = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    yield "gray_x2", big
    yield "clahe_x2", cv2.createCLAHE(2.0, (8, 8)).apply(big)
    yield "otsu_x2", cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    yield "rot90_x2", cv2.rotate(big, cv2.ROTATE_90_CLOCKWISE)


def decode_image(img_bgr: np.ndarray) -> list[dict]:
    """Return [{value, symbology, variant}] for the first variant that yields a non-empty result."""
    if not DECODER_AVAILABLE or img_bgr is None or img_bgr.size == 0:
        return []
    for name, v in _variants(img_bgr):
        res = [r for r in pyzbar.decode(v) if r.data]
        if res:
            out = []
            for r in res:
                try: val = r.data.decode("utf-8")
                except UnicodeDecodeError: val = r.data.decode("latin-1")
                out.append({"value": val.strip(), "symbology": r.type, "variant": name})
            return sorted(out, key=lambda d: LINEAR_FIRST.get(d["symbology"], 9))   # product barcodes before QR/other
    return []


def decode_detections(img_bgr: np.ndarray, detections: list[dict], pads=PADS, try_full_image: bool = True) -> list[dict]:
    """For each detection: crop (+padding) and decode. Adds a per-detection status:
    DECODED | DETECTED_NOT_DECODED | DECODER_UNAVAILABLE. Optionally falls back to the full image
    ONLY when no crop decoded (the result is labelled source='full_image')."""
    results = []
    for d in detections:
        item = {"box": d["box"], "det_conf": d["conf"], "decoded": [], "source": "crop"}
        if not DECODER_AVAILABLE:
            item["status"] = "DECODER_UNAVAILABLE"
        else:
            dec = []
            for pad in (pads if isinstance(pads, (tuple, list)) else (pads,)):
                dec = decode_image(crop_with_padding(img_bgr, d["box"], pad))
                if dec: break
            item["decoded"] = dec
            item["status"] = "DECODED" if dec else "DETECTED_NOT_DECODED"
        results.append(item)
    if try_full_image and DECODER_AVAILABLE and results and not any(r["status"] == "DECODED" for r in results):
        dec = decode_image(img_bgr)
        if dec:
            results[0].update(decoded=dec, status="DECODED", source="full_image")
    return results
