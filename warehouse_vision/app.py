"""Smart Warehouse Vision: Automated Product Identification and Inventory Verification (Streamlit)."""
from __future__ import annotations
import io
from pathlib import Path
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from src import config as C
from src.catalogue import load_catalogue
from src import decoder
from src.verification import VISUAL_ONLY

st.set_page_config(page_title="Smart Warehouse Vision", page_icon="📦", layout="wide")
MAX_BYTES = 20 * 1024 * 1024


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="Loading classifier…")
def get_classifier(path: str, mtime: float):
    """Returns ((model, class_names, device) | None, message). Never raises."""
    p = Path(path)
    if not p.is_file():
        return None, f"Classifier checkpoint not found: `{p}`. Train it with `python scripts/03_train_classifier.py --promote` (see README)."
    try:
        import torch
        from src.classifier import load_checkpoint
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, names = load_checkpoint(p, dev)
        return (model, names, dev), ""
    except ImportError as e:
        return None, f"PyTorch is not installed ({e}). `pip install -r requirements.txt`."
    except Exception as e:
        return None, f"Classifier checkpoint could not be loaded ({type(e).__name__}: {e}). Retrain with `scripts/03_train_classifier.py`."


@st.cache_resource(show_spinner="Loading barcode detector…")
def get_detector(path: str, mtime: float):
    try:
        from src.detector import load_detector
        return load_detector(Path(path))
    except Exception as e:
        return None, f"Detector unavailable ({type(e).__name__}: {e})"


@st.cache_data
def get_catalogue(path: str, mtime: float):
    return load_catalogue(Path(path))


def read_image(upload):
    """Validate and decode an upload. Returns (PIL image | None, error message)."""
    data = upload.getvalue()
    if len(data) > MAX_BYTES:
        return None, "File is larger than 20 MB."
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        img = Image.open(io.BytesIO(data)); img.load()
        return ImageOps.exif_transpose(img).convert("RGB"), ""
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None, "This file is not a valid or readable JPG/PNG image."


# ------------------------------------------------------------------ sidebar
st.title("📦 Smart Warehouse Vision")
st.caption("Automated Product Identification and Inventory Verification — evidence-aware prototype (not validated for commercial warehouse use)")

with st.sidebar:
    st.header("Settings")
    thr = st.slider("Visual-prediction confidence threshold", 0.05, 0.99, float(C.load_threshold()), 0.01,
                    help="Below this the top-1 class is flagged *uncertain*. Default comes from validation-set calibration (scripts/09). A probability is a model score, not a guarantee.")
    det_conf = st.slider("Barcode detection confidence", 0.05, 0.95, 0.25, 0.05)
    use_count = st.checkbox("Enter a reported count to compare with the catalogue")
    reported = int(st.number_input("Reported count", min_value=0, value=0, step=1)) if use_count else None
    cat_up = st.file_uploader("Optional: replace catalogue (CSV)", type=["csv"])
    st.divider()

clf, clf_msg = get_classifier(str(C.CLASSIFIER_CKPT), _mtime(C.CLASSIFIER_CKPT))
det, det_msg = get_detector(str(C.DETECTOR_WEIGHTS), _mtime(C.DETECTOR_WEIGHTS))
if cat_up is not None:
    catalogue, cat_msg = load_catalogue(io.BytesIO(cat_up.getvalue()))
else:
    catalogue, cat_msg = get_catalogue(str(C.CATALOGUE_CSV), _mtime(C.CATALOGUE_CSV))

with st.sidebar:
    st.subheader("Component status")
    st.write(("✅" if clf else "⚠️") + " Classifier (ResNet-18)")
    st.write(("✅" if det else "⚠️") + " Barcode detector (YOLOv8n)")
    st.write(("✅" if decoder.DECODER_AVAILABLE else "⚠️") + " Barcode decoder (pyzbar)")
    st.write(("✅" if catalogue else "⚠️") + f" Catalogue ({len(catalogue)} records)")
for m in (clf_msg, det_msg, cat_msg):
    if m: st.warning(m)
if not decoder.DECODER_AVAILABLE:
    st.warning(f"Barcode decoding disabled: {decoder.DECODER_ERROR}. Ubuntu: `sudo apt install -y libzbar0`")

# ------------------------------------------------------------------ main
upload = st.file_uploader("Upload a product / shelf image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if upload is None:
    st.info("Upload an image to run the pipeline. No new training data is needed. Each result below comes from a different source and is shown separately.")
    st.stop()

img, err = read_image(upload)
if img is None:
    st.error(err); st.stop()

from src.pipeline import run_pipeline
try:
    with st.spinner("Analysing…"):
        res = run_pipeline(img, clf, det, catalogue, thr, det_conf, reported)
except Exception as e:  # keep the app usable
    st.error(f"Analysis failed ({type(e).__name__}: {e}). Check the component status in the sidebar."); st.stop()

left, right = st.columns([1, 1])
left.subheader("Uploaded image"); left.image(img, use_container_width=True)
right.subheader("Barcode detections / decoded regions"); right.image(res["annotated"], use_container_width=True)
right.caption("Green = decoded, red = detected but not decoded. 'det' is DETECTION confidence, not decoding confidence.")

v = res["verification"]
st.divider(); st.header("Result summary")
status = v["status"]
{"Product found": st.success, "Inventory mismatch": st.warning, "Unknown barcode": st.warning, "Barcode unreadable": st.error,
 "Decoder unavailable": st.error, VISUAL_ONLY: st.info}.get(status, st.info)(f"**{status}** — {v['detail']}")
for f in v["flags"]:
    st.warning(f)

c1, c2 = st.columns(2)
with c1.container(border=True):
    st.subheader("1 · Vision model output"); st.caption("Source: trained ResNet-18 (fine-grained grocery class). Not a SKU/GTIN.")
    vis = res["visual"]
    if not vis["available"]:
        st.info("Classifier not loaded (see sidebar).")
    else:
        if vis["uncertain"]:
            st.warning(f"Uncertain — top-1 confidence {vis['top1'][1]:.0%} is below the {thr:.0%} threshold.")
        else:
            st.success(f"Top-1 above threshold ({thr:.0%}); still only a model score, not a guarantee.")
        for i, (name, p) in enumerate(vis["top3"], 1):
            st.write(f"**{i}. {name}** — {p:.1%}"); st.progress(min(max(p, 0.0), 1.0))
with c2.container(border=True):
    st.subheader("2 · Barcode detector output"); st.caption("Source: trained YOLOv8n; it only locates barcode regions.")
    if det is None:
        st.info("Detector not loaded (see sidebar); decoding falls back to the full image.")
    elif not res["detections"]:
        st.info("No barcode region detected at this confidence.")
    else:
        st.dataframe([{"#": i + 1, "box (x1,y1,x2,y2)": [int(x) for x in d["box"]], "detection confidence": round(d["conf"], 3)}
                      for i, d in enumerate(res["detections"])], hide_index=True, use_container_width=True)

c3, c4 = st.columns(2)
with c3.container(border=True):
    st.subheader("3 · Barcode decoder output"); st.caption("Source: pyzbar library (deterministic; not a trained model).")
    if res["decode_note"]: st.info(res["decode_note"])
    rows = [{"region": i + 1, "status": r["status"], "value": d["value"], "symbology": d["symbology"], "source": r["source"]}
            for i, r in enumerate(res["decode_results"]) for d in (r["decoded"] or [{"value": "—", "symbology": "—"}])]
    if rows: st.dataframe(rows, hide_index=True, use_container_width=True)
    else: st.info("Nothing to decode.")
with c4.container(border=True):
    st.subheader("4 · Catalogue lookup"); st.caption("Source: local catalogue CSV (records marked SIMULATED are demo data).")
    row = v["catalogue_row"]
    if row:
        st.write(f"**Barcode:** {v['barcode']}")
        st.dataframe([{k: row.get(k, "") for k in ("sku", "product_name", "category", "expected_quantity", "location", "status", "record_source")}],
                     hide_index=True, use_container_width=True)
    elif v["barcode"]:
        st.warning(f"No catalogue record for `{v['barcode']}`.")
    else:
        st.info("No decoded barcode to look up. The visual class is never used to look up a SKU.")

with st.container(border=True):
    st.subheader("5 · Inventory verification")
    st.write(f"**Status:** {status}")
    if reported is not None: st.write(f"Reported count: {reported}" + (f" · Expected: {row['expected_quantity']}" if row else ""))
    st.caption("Classifying one image does not count physical stock. A real stock count needs defined counting and duplicate-prevention logic.")
