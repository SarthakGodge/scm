"""Assemble outputs/reports/RESULTS.md from the JSON reports (so report numbers are never hand-typed)."""
import sys, json, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
R = C.REPORTS
ld = lambda p: json.loads(Path(p).read_text()) if Path(p).is_file() else None
L = ["# Results (auto-generated from outputs/reports/*.json)\n"]
f = ld(R / "classifier_test_metrics.json"); b = ld(C.OUT / "runs/baseline/classifier_test_metrics.json")
L += ["## Experiment 1 - ResNet-18 fine-grained classification (81 classes, official test split)\n", "| Model | Test acc | Macro P | Macro R | Macro F1 |", "|---|---|---|---|---|"]
if b: L.append(f"| Baseline (lr 1e-3, squash eval) | {b['test_accuracy']:.4f} | {b['macro_precision']:.4f} | {b['macro_recall']:.4f} | {b['macro_f1']:.4f} |")
if f: L.append(f"| **Final (tuned)** | **{f['test_accuracy']:.4f}** | **{f['macro_precision']:.4f}** | **{f['macro_recall']:.4f}** | **{f['macro_f1']:.4f}** |")
if f: L.append(f"\nImages: {f['num_test_images']} | inference {f['inference_ms_per_image_model_only']:.2f} ms/img ({f['inference_timing_note']}) | args: `{ {k: v for k, v in f['args'].items() if k in ('lr','aug','ls','dropout','img_size','cosine','mixup','iconic','eval_mode','seed')} }`")
tune = [ld(p) for p in sorted(glob.glob(str(C.OUT / "runs/t*/val_summary.json")))]
if tune:
    L += ["\n### Validation-only tuning (test set untouched)\n", "| Run | Val acc | Val loss |", "|---|---|---|"] + [f"| {d['run']} | {d['best_val_acc']:.4f} | {d['best_val_loss']:.4f} |" for d in tune]
d = ld(R / "detector_metrics.json")
if d:
    L += ["\n## Experiment 2 - YOLOv8n barcode-region detection (InventBar, single class)\n", f"imgsz {d['imgsz']}, batch {d['batch']}, seed {d['seed']}, ultralytics {d['ultralytics']}\n",
          "| Split | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |", "|---|---|---|---|---|"] + \
         [f"| {s} | {d[s]['precision']:.3f} | {d[s]['recall']:.3f} | {d[s]['mAP@0.50']:.3f} | {d[s]['mAP@0.50:0.95']:.3f} |" for s in ("val", "test")] + [f"\n{d['metric_notes']}"]
s = ld(R / "decode_study.json")
if s:
    L += ["\n## Barcode decoding (pyzbar, not a trained model) - decode RATE only\n", "| Method | Images decoded |", "|---|---|", f"| Ground-truth crop | {s['decode_rate_gt_crops']:.1%} |", f"| Full image | {s['decode_rate_full_image']:.1%} |"]
    if "decode_rate_detector_crops" in s: L.append(f"| YOLO detection crop (end-to-end) | {s['decode_rate_detector_crops']:.1%} |")
    L.append(f"\nn = {s['n_images']} test images; symbologies: {s['symbologies_decoded']}. {s['note']}.")
p = ld(C.POLICY_JSON)
if p:
    L += ["\n## Confidence policy (threshold chosen on validation)\n", f"Threshold **{p['threshold']}** - {p['rule']} (target met on val: {p['target_met_on_val']}).\n",
          "| Split | Coverage | Accuracy of confident predictions | ECE |", "|---|---|---|---|",
          f"| val | {p['val']['coverage']:.1%} | {p['val']['accepted_accuracy']:.1%} | {p['ece_val']:.3f} |", f"| test | {p['test']['coverage']:.1%} | {p['test']['accepted_accuracy']:.1%} | {p['ece_test']:.3f} |"]
(R / "RESULTS.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
