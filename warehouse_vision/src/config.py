"""Central configuration: paths, seeds, and hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GROCERY_ROOT = RAW / "GroceryStoreDataset"
GROCERY_DIR = GROCERY_ROOT / "dataset"          # manifest paths are relative to THIS dir
INVENTBAR_DIR = RAW / "BenchmarkBarcodeDatasets" / "InventBar"
PROCESSED = ROOT / "data" / "processed"
YOLO_DIR = PROCESSED / "inventbar_yolo"
OUT = ROOT / "outputs"
REPORTS, FIGURES, MODELS = OUT / "reports", OUT / "figures", OUT / "models"
CLASSIFIER_CKPT = MODELS / "resnet18_grocery_best.pt"
DETECTOR_WEIGHTS = OUT / "yolo_runs" / "barcode_yolov8n" / "weights" / "best.pt"
CATALOGUE_CSV = ROOT / "data" / "catalogue.csv"

SEED = 42
IMG_SIZE = 224
CLS_BATCH, CLS_EPOCHS, CLS_LR, CLS_PATIENCE = 32, 25, 1e-3, 5
YOLO_IMGSZ, YOLO_BATCH, YOLO_EPOCHS = 640, 16, 60
CONF_THRESHOLD = 0.60          # below this a visual prediction is flagged "uncertain"
for d in (REPORTS, FIGURES, MODELS, PROCESSED):
    d.mkdir(parents=True, exist_ok=True)

POLICY_JSON = REPORTS / "confidence_policy.json"


def load_threshold(default: float = CONF_THRESHOLD) -> float:
    """Threshold chosen on the VALIDATION split by scripts/09_calibrate_threshold.py (falls back to the default)."""
    import json
    try:
        return float(json.loads(POLICY_JSON.read_text())["threshold"])
    except Exception:
        return default
