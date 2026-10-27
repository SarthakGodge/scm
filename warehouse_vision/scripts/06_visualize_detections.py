"""Draw predicted barcode boxes (red) next to ground truth (green) on held-out test images."""
import sys, argparse, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config as C
from src.detector import load_detector, detect

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); ap.add_argument("--conf", type=float, default=0.25)
a = ap.parse_args()
model, msg = load_detector(C.DETECTOR_WEIGHTS)
if model is None: sys.exit(msg)
imgs = sorted((C.YOLO_DIR / "images" / "test").glob("*.jpg")); random.Random(0).shuffle(imgs)
fig, axes = plt.subplots(2, 4, figsize=(16, 9))
for ax, p in zip(axes.ravel(), imgs[:a.n]):
    im = cv2.imread(str(p)); H, W = im.shape[:2]
    for l in (C.YOLO_DIR / "labels" / "test" / (p.stem + ".txt")).read_text().split("\n"):
        if l.strip():
            _, x, y, w, h = map(float, l.split()); cv2.rectangle(im, (int((x-w/2)*W), int((y-h/2)*H)), (int((x+w/2)*W), int((y+h/2)*H)), (0, 200, 0), 3)
    for d in detect(model, cv2.imread(str(p)), a.conf):
        x1, y1, x2, y2 = map(int, d["box"]); cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(im, f"{d['conf']:.2f}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.axis("off"); ax.set_title(p.name, fontsize=8)
fig.suptitle("green = ground truth, red = predicted (detection confidence)"); plt.tight_layout()
plt.savefig(C.FIGURES / "detector_predictions.png", dpi=110); print("Saved", C.FIGURES / "detector_predictions.png")
