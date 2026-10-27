"""Evidence-aware verification. Keeps five kinds of evidence separate and never merges them into one
'identity' without support:
  visual prediction | barcode detection | barcode decoding | catalogue lookup | inventory verification"""
from __future__ import annotations
from .config import CONF_THRESHOLD
from .catalogue import lookup

VISUAL_ONLY = "Visual prediction only — SKU not verified"


def assess_visual(top_preds: list[tuple[str, float]] | None, threshold: float = CONF_THRESHOLD) -> dict:
    """top_preds = [(class_name, prob), ...] sorted desc. Low confidence => 'uncertain'.
    A probability is a model score, not a guarantee of correctness."""
    if not top_preds:
        return {"available": False, "uncertain": None, "top1": None, "top3": []}
    return {"available": True, "top1": top_preds[0], "top3": top_preds[:3], "threshold": threshold,
            "uncertain": top_preds[0][1] < threshold}


def verify(visual: dict, decode_results: list[dict] | None, catalogue: dict, reported_count: int | None = None,
           detector_available: bool = True) -> dict:
    """Returns {'status': str, 'detail': str, 'flags': [..], 'catalogue_row': dict|None, 'barcode': str|None}."""
    flags: list[str] = []
    if visual.get("available") and visual["uncertain"]:
        flags.append(f"Visual prediction uncertain (top-1 {visual['top1'][1]:.0%} < threshold {visual['threshold']:.0%})")

    decode_results = decode_results or []
    decoded = [(r, d) for r in decode_results for d in r["decoded"]]

    if not decoded:
        if decode_results and all(r.get("status") == "DECODER_UNAVAILABLE" for r in decode_results):
            return {"status": "Decoder unavailable", "barcode": None, "catalogue_row": None, "flags": flags,
                    "detail": "A barcode region was detected, but the decoder library (pyzbar/libzbar) is not available. Ubuntu: sudo apt install -y libzbar0"}
        if decode_results:  # barcodes located but nothing decoded
            return {"status": "Barcode unreadable", "barcode": None, "catalogue_row": None, "flags": flags,
                    "detail": "A barcode region was detected but could not be decoded. Retake the photo closer/sharper or enter the code manually."}
        if visual.get("available"):
            why = "" if detector_available else " (barcode detector not loaded)"
            return {"status": VISUAL_ONLY, "barcode": None, "catalogue_row": None, "flags": flags,
                    "detail": f"No barcode evidence{why}. The class is a model prediction only; it is not a SKU/GTIN."}
        return {"status": "No evidence", "barcode": None, "catalogue_row": None, "flags": flags,
                "detail": "No classifier output and no barcode found."}

    # try every decoded value against the catalogue (e.g. a QR URL may sit beside the product barcode); else use the first
    code, row = decoded[0][1]["value"], None
    for _, d in decoded:
        hit = lookup(catalogue, d["value"])
        if hit is not None: code, row = d["value"], hit; break
    if row is None:
        return {"status": "Unknown barcode", "barcode": code, "catalogue_row": None, "flags": flags,
                "detail": f"Decoded '{code}' but it is not in the local catalogue."}

    if row.get("status", "").lower() not in ("", "active"):
        flags.append(f"Catalogue status is '{row['status']}'")
    if row.get("record_source", "").upper() == "SIMULATED":
        flags.append("Catalogue record is SIMULATED (demo data)")
    mapped = row.get("visual_class", "").strip()
    if mapped and visual.get("available") and not visual["uncertain"] and visual["top1"][0] != mapped:
        flags.append(f"Conflicting evidence: catalogue maps this barcode to '{mapped}' but the image classifier predicts '{visual['top1'][0]}'")

    if reported_count is not None and str(row["expected_quantity"]).strip().lstrip("-").isdigit():
        exp = int(row["expected_quantity"])
        if reported_count != exp:
            return {"status": "Inventory mismatch", "barcode": code, "catalogue_row": row, "flags": flags,
                    "detail": f"Reported count {reported_count} differs from expected {exp}."}
    return {"status": "Product found", "barcode": code, "catalogue_row": row, "flags": flags,
            "detail": "Barcode decoded and found in the catalogue. This does not count physical stock; a real count needs defined counting and duplicate-prevention logic."}
