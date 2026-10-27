#!/usr/bin/env bash
# Tuning on VALIDATION only (--skip-test). Compare best_val_acc / best_val_loss in outputs/runs/*/val_summary.json
set -e; cd "$(dirname "$0")/.."; unset LD_LIBRARY_PATH
# t0 = the recipe from your original train.py (lr 1e-4, crop eval, basic aug, 15 epochs)
python scripts/03_train_classifier.py --run-name t0_prev_recipe --skip-test
R="--cosine --aug strong --ls 0.1 --dropout 0.2 --lr 3e-4 --epochs 30 --patience 10 --skip-test"
python scripts/03_train_classifier.py --run-name t1_regularised  $R
python scripts/03_train_classifier.py --run-name t2_plus_iconic  $R --iconic
python scripts/03_train_classifier.py --run-name t3_plus_mixup   $R --mixup 0.2
python scripts/03_train_classifier.py --run-name t4_img288       $R --img-size 288
python - <<'P'
import json,glob
for f in sorted(glob.glob("outputs/runs/*/val_summary.json")):
    d=json.load(open(f)); print(f"{d['run']:22s} val_acc={d['best_val_acc']:.4f} val_loss={d['best_val_loss']:.4f}")
P
