"""End-to-end evidence pipeline used by the Streamlit app. Each stage is optional and reported separately."""
from __future__ import annotations
import numpy as np, cv2
from PIL import Image
from . import decoder, config as C
from .detector import detect
from .verification import assess_visual, verify

MAX_SIDE = 2560


def classify(model, class_names, pil_img: Image.Image, device="cpu", k: int = 3):
    """Top-k [(class_name, probability)] using the preprocessing stored with the checkpoint."""
    import torch
    from .classifier import eval_tf
    tf = eval_tf(getattr(model, "img_size", C.IMG_SIZE), getattr(model, "eval_mode", "crop"))
    x = tf(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    s, i = torch.topk(p, min(k, len(class_names)))
    return [(class_names[j], float(v)) for v, j in zip(s.tolist(), i.tolist())]


def draw_boxes(img_bgr, results):
    out = img_bgr.copy(); t = max(2, round(max(out.shape[:2]) / 300))
    for r in results:
        if r.get("box") is None: continue
        x1, y1, x2, y2 = map(int, r["box"]); color = (0, 170, 0) if r["status"] == "DECODED" else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
        cv2.putText(out, f"det {r['det_conf']:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, t / 3), color, max(1, t - 1))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def run_pipeline(pil_img: Image.Image, clf=None, detector=None, catalogue=None, conf_threshold=None, det_conf=0.25,
                 reported_count=None) -> dict:
    """clf = (model, class_names, device) or None; detector = YOLO or None."""
    conf_threshold = C.load_threshold() if conf_threshold is None else conf_threshold
    pil_img = pil_img.convert("RGB")
    if max(pil_img.size) > MAX_SIDE:
        pil_img.thumbnail((MAX_SIDE, MAX_SIDE))
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    top = classify(clf[0], clf[1], pil_img, clf[2]) if clf else None
    visual = assess_visual(top, conf_threshold)

    detections = detect(detector, bgr, det_conf) if detector is not None else []
    results = decoder.decode_detections(bgr, detections) if detections else []
    note = ""
    if not any(r["status"] == "DECODED" for r in results) and decoder.DECODER_AVAILABLE:
        full = decoder.decode_image(bgr)           # fallback, clearly labelled
        if full:
            note = ("Detector not loaded; decoded from the full image." if detector is None else
                    "Detector found no decodable region; decoded from the full image instead.")
            results = [{"box": None, "det_conf": None, "decoded": full, "source": "full_image", "status": "DECODED"}] + [r for r in results]
    verification = verify(visual, results, catalogue or {}, reported_count, detector_available=detector is not None)
    return {"visual": visual, "detections": detections, "decode_results": results, "decode_note": note,
            "verification": verification, "annotated": draw_boxes(bgr, results)}
