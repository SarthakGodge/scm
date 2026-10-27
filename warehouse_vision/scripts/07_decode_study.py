"""Measure barcode DECODING separately from detection (pyzbar is a library, not a trained model).
 (a) ground-truth boxes -> crop(+pad) -> decode   : decoder-only performance (no detector needed)
 (b) full image -> decode                          : baseline without localisation
 (c) YOLO detections -> crop -> decode (if weights exist): end-to-end detect+decode
IMPORTANT: InventBar has no ground-truth barcode text, so these are DECODE RATES (a value was returned),
not correctness rates. pyzbar validates EAN/UPC check digits, which makes wrong reads less likely, not impossible."""
import sys, json, argparse, csv
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from src import config as C
from src import decoder
from src.detector import load_detector, detect

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test"); ap.add_argument("--add-to-catalogue", type=int, default=0,
                help="append N decoded values to data/catalogue.csv as SIMULATED demo items (so the live demo can show 'Product found')")
a = ap.parse_args()
if not decoder.DECODER_AVAILABLE:
    sys.exit(f"pyzbar unavailable: {decoder.DECODER_ERROR}\nUbuntu: sudo apt install -y libzbar0")
imgs = sorted((C.YOLO_DIR / "images" / a.split).glob("*.jpg")); assert imgs, "run 02_prepare_inventbar.py first"
det, msg = load_detector(C.DETECTOR_WEIGHTS)
if det is None: print("NOTE:", msg, "- skipping (c)")
S = {"gt_crop": 0, "full_image": 0, "detect_decode": 0}; sym = Counter(); vals = {}; n_det = 0
for p in imgs:
    im = cv2.imread(str(p)); H, W = im.shape[:2]
    lab = [l.split() for l in (C.YOLO_DIR / "labels" / a.split / (p.stem + ".txt")).read_text().splitlines() if l.strip()]
    gt = [[(float(x)-float(w)/2)*W, (float(y)-float(h)/2)*H, (float(x)+float(w)/2)*W, (float(y)+float(h)/2)*H] for _, x, y, w, h in lab]
    r = decoder.decode_detections(im, [{"box": b, "conf": 1.0} for b in gt], try_full_image=False)
    if any(x["status"] == "DECODED" for x in r):
        S["gt_crop"] += 1
        for d in r[0]["decoded"]: sym[d["symbology"]] += 1; vals[p.name] = d["value"]
    if decoder.decode_image(im): S["full_image"] += 1
    if det is not None:
        ds = detect(det, im); n_det += bool(ds)
        if any(x["status"] == "DECODED" for x in decoder.decode_detections(im, ds, try_full_image=False)): S["detect_decode"] += 1
n = len(imgs)
rep = {"split": a.split, "n_images": n, "decode_rate_gt_crops": S["gt_crop"] / n, "decode_rate_full_image": S["full_image"] / n,
       "symbologies_decoded": dict(sym), "note": "decode RATE only; no ground-truth text exists to verify correctness"}
if det is not None:
    rep.update(images_with_detection=n_det / n, decode_rate_detector_crops=S["detect_decode"] / n)
(C.REPORTS / "decode_study.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))
if a.add_to_catalogue:
    cat = C.CATALOGUE_CSV; existing = {r["barcode"] for r in csv.DictReader(open(cat))}
    new = [v for v in dict.fromkeys(vals.values()) if v not in existing][:a.add_to_catalogue]
    with open(cat, "a", newline="") as f:
        w = csv.writer(f)
        for i, v in enumerate(new, 1):
            w.writerow([v, f"SKU-DEMO-{100+i:04d}", f"Demo Item {100+i} (simulated)", "Demo Category (simulated)", 10 + i, f"D-{i:02d}-01", "active", "", "SIMULATED"])
    print(f"Added {len(new)} SIMULATED demo records to {cat}")
