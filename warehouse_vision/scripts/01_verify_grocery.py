"""Verify GroceryStoreDataset: parse manifests, check files/labels, plot samples, write report."""
import sys, json, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from src.config import REPORTS, FIGURES, SEED
from src.grocery_data import load_classes, parse_manifest, SPLITS

classes = load_classes()
ids = sorted(classes)
assert ids == list(range(len(ids))), "class IDs are not contiguous 0..N-1"
print(f"Classes: {len(classes)} (ids {ids[0]}..{ids[-1]})")

summary, all_records = {"num_classes": len(classes)}, {}
for s in SPLITS:
    recs, rep, problems = parse_manifest(s, classes=classes, check_open=True)
    all_records[s] = recs
    summary[s] = rep
    print(f"[{s}] {rep}")
    for p in problems[:5]:
        print("   problem:", p)
    assert max(l for _, l in recs) < len(classes) and min(l for _, l in recs) >= 0

# leakage check
sets = {s: {p for p, _ in all_records[s]} for s in SPLITS}
summary["split_overlap"] = {f"{a}&{b}": len(sets[a] & sets[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
print("Split overlap:", summary["split_overlap"])
(REPORTS / "grocery_verification.json").write_text(json.dumps(summary, indent=2))

random.seed(SEED)
sample = random.sample(all_records["train"], 8)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, (p, l) in zip(axes.ravel(), sample):
    ax.imshow(Image.open(p).convert("RGB")); ax.set_title(f"{l}: {classes[l][0]}", fontsize=8); ax.axis("off")
plt.tight_layout(); plt.savefig(FIGURES / "grocery_samples.png", dpi=120)
print("Saved report + sample figure. Grocery data verified OK.")
