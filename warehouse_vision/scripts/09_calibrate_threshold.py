"""Choose the 'uncertain' confidence threshold on the VALIDATION split, then report its effect on TEST (once).
Rule: smallest threshold whose accepted (confident) validation predictions reach --target accuracy, keeping coverage >= --min-coverage.
A softmax probability is a model score, not a guarantee of correctness; ECE is reported to show how far it is from calibrated."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from torch.utils.data import DataLoader
from src import config as C
from src.grocery_data import load_classes, parse_manifest
from src.classifier import GroceryDS, eval_tf, load_checkpoint

ap = argparse.ArgumentParser(); ap.add_argument("--target", type=float, default=0.95); ap.add_argument("--min-coverage", type=float, default=0.5)
ap.add_argument("--ckpt", type=Path, default=C.CLASSIFIER_CKPT); a = ap.parse_args()
if not a.ckpt.is_file(): sys.exit(f"{a.ckpt} missing. Train first: python scripts/03_train_classifier.py --promote")
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, names = load_checkpoint(a.ckpt, dev); classes = load_classes()


@torch.no_grad()
def probs(split):
    recs = parse_manifest(split, classes=classes)[0]
    dl = DataLoader(GroceryDS(recs, eval_tf(model.img_size, model.eval_mode)), batch_size=64, num_workers=4)
    P, Y = [], []
    for x, y in dl: P.append(torch.softmax(model(x.to(dev)), 1).cpu()); Y += y.tolist()
    P = torch.cat(P).numpy(); return P.max(1), P.argmax(1) == np.array(Y)


def ece(conf, ok, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def stats(conf, ok, t):
    m = conf >= t; return {"coverage": float(m.mean()), "accepted_accuracy": float(ok[m].mean()) if m.any() else None,
                           "n_accepted": int(m.sum()), "uncertain_error_rate": float((~ok[~m]).mean()) if (~m).any() else None}


cv, okv = probs("val"); ct, okt = probs("test")
grid = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]
table = [{"threshold": t, **stats(cv, okv, t)} for t in grid]
good = [r for r in table if r["accepted_accuracy"] is not None and r["accepted_accuracy"] >= a.target and r["coverage"] >= a.min_coverage]
chosen = good[0] if good else max((r for r in table if r["accepted_accuracy"] is not None), key=lambda r: (r["accepted_accuracy"], r["coverage"]))
t = chosen["threshold"]
pol = {"threshold": t, "rule": f"smallest threshold with validation accepted-accuracy >= {a.target} and coverage >= {a.min_coverage}",
       "target_met_on_val": bool(good), "val": stats(cv, okv, t), "test": stats(ct, okt, t),
       "overall_val_accuracy": float(okv.mean()), "overall_test_accuracy": float(okt.mean()),
       "ece_val": ece(cv, okv), "ece_test": ece(ct, okt), "val_table": table,
       "note": "Probability is a model score, not a guarantee of correctness. Threshold is configurable in the app."}
C.POLICY_JSON.write_text(json.dumps(pol, indent=2))
print(f"Chosen threshold {t} (target met on val: {bool(good)})\n VAL  {pol['val']}\n TEST {pol['test']}\n ECE val {pol['ece_val']:.3f} test {pol['ece_test']:.3f}\nSaved {C.POLICY_JSON}")
