"""Unit tests for the verification rules (no GPU / no trained model needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.catalogue import load_catalogue
from src.verification import assess_visual, verify, VISUAL_ONLY
from src import config as C
cat, w = load_catalogue(C.CATALOGUE_CSV); assert cat, w
dec = lambda v: [{"box": [0,0,1,1], "det_conf": .9, "decoded": [{"value": v, "symbology": "EAN13", "variant": "gray"}], "status": "DECODED"}]
nodec = [{"box": [0,0,1,1], "det_conf": .8, "decoded": [], "status": "DETECTED_NOT_DECODED"}]
vis_hi = assess_visual([("Orange", .95), ("Lemon", .03), ("Lime", .01)]); vis_lo = assess_visual([("Orange", .40), ("Lemon", .30), ("Lime", .1)])
T = [("found",        verify(vis_hi, dec("5901234123457"), cat),            "Product found"),
     ("unknown",      verify(vis_hi, dec("0000000000017"), cat),            "Unknown barcode"),
     ("unreadable",   verify(vis_hi, nodec, cat),                           "Barcode unreadable"),
     ("mismatch",     verify(vis_hi, dec("5901234123457"), cat, 5),         "Inventory mismatch"),
     ("count ok",     verify(vis_hi, dec("5901234123457"), cat, 24),        "Product found"),
     ("visual only",  verify(vis_hi, [], cat),                              VISUAL_ONLY),
     ("no evidence",  verify({"available": False}, [], cat),                "No evidence"),
     ("UPC-A=EAN13",  verify(vis_hi, dec("0012345678905"), cat),            "Product found")]
ok = True
for name, r, exp in T:
    good = r["status"] == exp; ok &= good; print(("PASS" if good else "FAIL"), name, "->", r["status"])
assert any("uncertain" in f for f in verify(vis_lo, [], cat)["flags"]), "low confidence must be flagged"
assert any("SIMULATED" in f for f in verify(vis_hi, dec("5901234123457"), cat)["flags"])
assert any("discontinued" in f for f in verify(vis_hi, dec("012345678905"), cat)["flags"])
cat2 = {k: dict(v) for k, v in cat.items()}; cat2["5901234123457"]["visual_class"] = "Lemon"
assert any("Conflicting" in f for f in verify(vis_hi, dec("5901234123457"), cat2)["flags"])
qr_first = [{"box": [0,0,1,1], "det_conf": .9, "status": "DECODED", "decoded": [{"value": "https://x.y/z", "symbology": "QRCODE", "variant": "o"}, {"value": "5901234123457", "symbology": "EAN13", "variant": "o"}]}]
assert verify(vis_hi, qr_first, cat)["status"] == "Product found", "must find the catalogue match even if a QR code is listed first"
na = [{"box": [0,0,1,1], "det_conf": .8, "decoded": [], "status": "DECODER_UNAVAILABLE"}]
assert verify(vis_hi, na, cat)["status"] == "Decoder unavailable"
print("PASS QR-beside-barcode handling + decoder-unavailable status")
print("PASS low-confidence flag / simulated flag / discontinued flag / conflicting-evidence flag")
sys.exit(0 if ok else 1)
