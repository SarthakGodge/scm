"""Fine-tune YOLOv8n (COCO-pretrained) to detect barcode regions on InventBar; evaluate on the held-out TEST split once."""
import sys, json, argparse, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=C.YOLO_EPOCHS)
ap.add_argument("--batch", type=int, default=C.YOLO_BATCH)
ap.add_argument("--imgsz", type=int, default=C.YOLO_IMGSZ)
ap.add_argument("--model", default="yolov8n.pt")
ap.add_argument("--patience", type=int, default=20)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=C.SEED)
a = ap.parse_args()

data_yaml = C.YOLO_DIR / "data.yaml"
if not data_yaml.is_file():
    sys.exit("Run scripts/02_prepare_inventbar.py first (data.yaml missing).")
# --- verify image/label pairing before training
for s in ("train", "val", "test"):
    imgs = sorted((C.YOLO_DIR / "images" / s).glob("*.jpg"))
    labs = [C.YOLO_DIR / "labels" / s / (p.stem + ".txt") for p in imgs]
    bad = [l.name for l in labs if not l.is_file() or l.stat().st_size == 0]
    if not imgs or bad:
        sys.exit(f"Split '{s}': {len(imgs)} images, bad/missing labels: {bad[:5]}. Refusing to train.")
    print(f"[{s}] {len(imgs)} images, labels OK")

import ultralytics, torch
from ultralytics import YOLO
project = (C.OUT / "yolo_runs").resolve()
model = YOLO(a.model)          # downloads pretrained weights automatically
model.train(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed, deterministic=True,
            patience=a.patience, workers=a.workers, project=str(project), name="barcode_yolov8n", exist_ok=True,
            device=0 if torch.cuda.is_available() else "cpu", plots=True)

best = project / "barcode_yolov8n" / "weights" / "best.pt"
final = YOLO(str(best))
dev = 0 if torch.cuda.is_available() else "cpu"
res = {}
for split in ("val", "test"):   # test is evaluated once, with the weights chosen on val
    m = final.val(data=str(data_yaml), split=split, imgsz=a.imgsz, batch=a.batch, device=dev, plots=(split == "test"),
                  project=str(project), name=f"eval_{split}", exist_ok=True, conf=0.001, iou=0.6)
    res[split] = {"precision": float(m.box.mp), "recall": float(m.box.mr),
                  "mAP@0.50": float(m.box.map50), "mAP@0.50:0.95": float(m.box.map)}
report = {"model": a.model, "ultralytics": ultralytics.__version__, "torch": torch.__version__, "python": platform.python_version(),
          "imgsz": a.imgsz, "batch": a.batch, "epochs_requested": a.epochs, "seed": a.seed, "task": "single-class barcode REGION detection",
          "metric_notes": "P/R at the F1-optimal confidence reported by Ultralytics; mAP at IoU 0.50 and averaged over 0.50:0.95; eval conf=0.001, NMS iou=0.6",
          "weights": str(best), **res}
(C.REPORTS / "detector_metrics.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
