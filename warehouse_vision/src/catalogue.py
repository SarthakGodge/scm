"""Local DEMONSTRATION catalogue (CSV). Records are simulated unless you add real ones.
Optional column `visual_class` is an EXPLICIT mapping to a classifier label; it is empty by default,
because the grocery class labels are NOT barcodes/SKUs."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

COLUMNS = ["barcode", "sku", "product_name", "category", "expected_quantity", "location", "status", "visual_class", "record_source"]


def normalise(code: str) -> str:
    return "".join(ch for ch in str(code).strip() if ch.isalnum()).upper()


def load_catalogue(path: Path) -> tuple[dict, str]:
    """Returns ({normalised_barcode: row_dict}, warning). Missing/bad file -> empty catalogue + warning."""
    if isinstance(path, (str, Path)):
        path = Path(path)
        if not path.is_file():
            return {}, f"Catalogue not found at {path}."
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        return {}, f"Could not read catalogue: {e}"
    if "barcode" not in df.columns:
        return {}, "Catalogue CSV needs a 'barcode' column."
    for c in COLUMNS:
        if c not in df.columns: df[c] = ""
    return {normalise(r["barcode"]): r.to_dict() for _, r in df.iterrows() if r["barcode"].strip()}, ""


def lookup(cat: dict, code: str):
    """Exact match on normalised value; also tries UPC-A <-> EAN-13 (leading zero) equivalence."""
    k = normalise(code)
    for cand in (k, "0" + k, k.lstrip("0")):
        if cand in cat: return cat[cand]
    for ck, row in cat.items():
        if ck.lstrip("0") == k.lstrip("0") and k.lstrip("0"): return row
    return None
