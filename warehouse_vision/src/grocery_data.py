"""Robust parsing of GroceryStoreDataset manifests (comma-separated: path, fine_id, coarse_id)."""
import csv, re
from collections import Counter
from pathlib import Path
from PIL import Image
from .config import GROCERY_DIR

SPLITS = ("train", "val", "test")


def load_classes(dataset_dir: Path = GROCERY_DIR):
    """Return {fine_id: (class_name, coarse_name)} from classes.csv (handles CRLF)."""
    p = dataset_dir / "classes.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run: git clone https://github.com/marcusklasson/GroceryStoreDataset data/raw/GroceryStoreDataset")
    classes = {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        next(rd)  # header
        for r in rd:
            if len(r) >= 3 and r[1].strip().isdigit():
                classes[int(r[1])] = (r[0].strip(), r[2].strip())
    if not classes:
        raise ValueError("classes.csv parsed to zero classes")
    return classes


def parse_manifest(split: str, dataset_dir: Path = GROCERY_DIR, classes=None, check_open=False):
    """Parse one split. Returns (records, report). records = [(abs_path, fine_id)]."""
    classes = classes or load_classes(dataset_dir)
    mpath = dataset_dir / f"{split}.txt"
    if not mpath.exists():
        raise FileNotFoundError(mpath)
    rep = Counter()
    records, seen = [], set()
    problems = []
    for ln, raw in enumerate(mpath.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        rep["lines"] += 1
        parts = [p.strip() for p in re.split(r"\s*,\s*", raw.strip())]
        if len(parts) < 2 or not parts[1].lstrip("-").isdigit():
            rep["malformed"] += 1; problems.append((ln, "malformed", raw)); continue
        rel, label = parts[0], int(parts[1])
        if label not in classes:
            rep["invalid_label"] += 1; problems.append((ln, "invalid_label", raw)); continue
        if rel in seen:
            rep["duplicate"] += 1; continue
        seen.add(rel)
        path = (dataset_dir / rel)
        if not path.is_file():
            rep["missing_file"] += 1; problems.append((ln, "missing_file", rel)); continue
        if check_open:
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception as e:
                rep["unreadable"] += 1; problems.append((ln, f"unreadable:{e}", rel)); continue
        # cross-check: folder name must equal class name from classes.csv
        if path.parent.name != classes[label][0]:
            rep["label_folder_mismatch"] += 1; problems.append((ln, "label_folder_mismatch", raw)); continue
        records.append((path, label))
    rep["usable"] = len(records)
    if not records:
        raise RuntimeError(f"No usable records in {split}. Report: {dict(rep)}; first problems: {problems[:5]}")
    return records, dict(rep), problems
