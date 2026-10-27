"""Validate InventBar YOLO annotations, make seeded train/val/test split, write data.yaml, plot samples."""
import sys, json, random, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as patches
from PIL import Image
from src.config import INVENTBAR_DIR, YOLO_DIR, REPORTS, FIGURES, SEED

if not INVENTBAR_DIR.exists():
    sys.exit(f"{INVENTBAR_DIR} missing. Run: git clone https://github.com/ChommaphatMa/BenchmarkBarcodeDatasets data/raw/BenchmarkBarcodeDatasets")

valid, failures, n_boxes = [], [], 0
for img in sorted(INVENTBAR_DIR.glob("*.jpg")):
    ann = img.with_suffix(".txt")
    if not ann.is_file():
        failures.append((img.name, "missing annotation")); continue
    lines = [l.split() for l in ann.read_text().splitlines() if l.strip()]
    if not lines:
        failures.append((img.name, "empty annotation")); continue   # dataset is barcode-only; empty = suspicious
    try:
        with Image.open(img) as im:
            im.verify()
        boxes, ok = [], True
        for t in lines:
            if len(t) != 5:
                raise ValueError(f"expected 5 fields, got {len(t)}")
            c = int(t[0]); x, y, w, h = map(float, t[1:])
            if c != 0: raise ValueError(f"unexpected class {c}")
            if w <= 0 or h <= 0: raise ValueError("non-positive size")
            if not (0 <= x - w/2 and x + w/2 <= 1.0001 and 0 <= y - h/2 and y + h/2 <= 1.0001):
                raise ValueError(f"box outside image: {x, y, w, h}")
            boxes.append((c, x, y, w, h))
        valid.append((img, boxes)); n_boxes += len(boxes)
    except Exception as e:
        failures.append((img.name, str(e)))

print(f"Valid images: {len(valid)}  boxes: {n_boxes}  failures: {len(failures)}")
for f in failures[:10]: print("  FAIL", f)
if not valid or len(failures) > 0.05 * (len(valid) + len(failures)):
    sys.exit("Too many invalid annotations - refusing to prepare dataset for training.")

random.Random(SEED).shuffle(valid)
n = len(valid); n_tr, n_va = int(0.70 * n), int(0.15 * n)
splits = {"train": valid[:n_tr], "val": valid[n_tr:n_tr + n_va], "test": valid[n_tr + n_va:]}
if YOLO_DIR.exists(): shutil.rmtree(YOLO_DIR)
for s, items in splits.items():
    (YOLO_DIR / "images" / s).mkdir(parents=True); (YOLO_DIR / "labels" / s).mkdir(parents=True)
    for img, boxes in items:
        shutil.copy(img, YOLO_DIR / "images" / s / img.name)
        (YOLO_DIR / "labels" / s / (img.stem + ".txt")).write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes) + "\n")
(YOLO_DIR / "data.yaml").write_text(
    f"path: {YOLO_DIR.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: barcode\n")

report = {"source": str(INVENTBAR_DIR), "annotation_format": "YOLO normalized (class xc yc w h), already native; no conversion needed",
          "seed": SEED, "split_sizes": {k: len(v) for k, v in splits.items()}, "total_boxes": n_boxes,
          "failures": failures,
          "limitations": "One box per image, single class; scenes with multiple barcodes may have unlabeled barcodes."}
(REPORTS / "inventbar_conversion.json").write_text(json.dumps(report, indent=2))
print("Splits:", report["split_sizes"])

fig, axes = plt.subplots(2, 4, figsize=(14, 8))
for ax, (img, boxes) in zip(axes.ravel(), random.Random(1).sample(valid, 8)):
    im = Image.open(img); W, H = im.size; ax.imshow(im); ax.axis("off"); ax.set_title(img.name, fontsize=8)
    for _, x, y, w, h in boxes:
        ax.add_patch(patches.Rectangle(((x - w/2) * W, (y - h/2) * H), w * W, h * H, fill=False, ec="lime", lw=2))
plt.tight_layout(); plt.savefig(FIGURES / "inventbar_boxes.png", dpi=110)
print("InventBar prepared OK ->", YOLO_DIR)
