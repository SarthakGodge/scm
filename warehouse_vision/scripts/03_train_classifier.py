"""Train an ImageNet-pretrained ResNet on GroceryStoreDataset.
Every run writes to outputs/runs/<run-name>/. Use --skip-test while tuning (compare on VAL only);
evaluate the test split once, for the final configuration. --promote copies the checkpoint to the app's path."""
import sys, json, time, argparse, random, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from src import config as C
from src.grocery_data import load_classes, parse_manifest
from src.classifier import GroceryDS, train_tf, eval_tf, build_model

ap = argparse.ArgumentParser()
ap.add_argument("--run-name", default="prev_recipe")
ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch", type=int, default=C.CLS_BATCH)
ap.add_argument("--lr", type=float, default=1e-4, help="head LR (and backbone LR unless --diff-lr)")
ap.add_argument("--backbone-lr", type=float, default=1e-4, help="used only with --diff-lr")
ap.add_argument("--diff-lr", action="store_true")
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--cosine", action="store_true", help="1 warm-up epoch + cosine decay")
ap.add_argument("--aug", choices=["basic", "strong"], default="basic")
ap.add_argument("--ls", type=float, default=0.0, help="label smoothing")
ap.add_argument("--dropout", type=float, default=0.0)
ap.add_argument("--mixup", type=float, default=0.0, help="mixup alpha (0=off)")
ap.add_argument("--img-size", type=int, default=C.IMG_SIZE)
ap.add_argument("--eval-mode", choices=["crop", "squash"], default="crop", help="crop = Resize256+CenterCrop224 (as train.py)")
ap.add_argument("--iconic", action="store_true", help="add the dataset's iconic images to TRAIN only")
ap.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation")
ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--seed", type=int, default=C.SEED)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--subset", type=int, default=0)
ap.add_argument("--skip-test", action="store_true")
ap.add_argument("--promote", action="store_true")
a = ap.parse_args()

RUN = C.OUT / "runs" / a.run_name; RUN.mkdir(parents=True, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:", dev, "| run:", a.run_name)

classes = load_classes(); names = [classes[i][0] for i in range(len(classes))]
recs = {s: parse_manifest(s, classes=classes, check_open=True)[0] for s in ("train", "val", "test")}
if a.iconic:
    extra = []
    for fid, row in classes.items():
        pass
    import csv
    with open(C.GROCERY_DIR / "classes.csv", newline="", encoding="utf-8-sig") as f:
        rd = csv.reader(f); next(rd)
        for r in rd:
            p = C.GROCERY_DIR / r[4].strip().lstrip("/")
            if p.is_file(): extra.append((p, int(r[1])))
    print(f"Adding {len(extra)} iconic images to train"); recs["train"] = recs["train"] + extra
if a.subset: recs["train"] = random.sample(recs["train"], a.subset)
g = torch.Generator(); g.manual_seed(a.seed)
mk = lambda s, tf, sh: DataLoader(GroceryDS(recs[s], tf), batch_size=a.batch, shuffle=sh, num_workers=a.workers,
                                   pin_memory=dev.type == "cuda", generator=g if sh else None)
tr = mk("train", train_tf(a.img_size, a.aug == "strong"), True)
va = mk("val", eval_tf(a.img_size, a.eval_mode), False); te = mk("test", eval_tf(a.img_size, a.eval_mode), False)

model = build_model(len(names), arch=a.arch, dropout=a.dropout).to(dev)
crit = nn.CrossEntropyLoss(label_smoothing=a.ls); val_crit = nn.CrossEntropyLoss()  # plain CE so val loss is comparable across runs
head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
back = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
groups = [{"params": back, "lr": a.backbone_lr if a.diff_lr else a.lr}, {"params": head, "lr": a.lr}]
opt = torch.optim.AdamW(groups, weight_decay=a.wd)
if a.cosine:
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=1)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs - 1))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[1]); plateau = False
else:
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2); plateau = True  # monitors val ACCURACY


def train_epoch():
    model.train(); tot = cor = 0; ls = 0.0
    for x, y in tr:
        x, y = x.to(dev), y.to(dev)
        if a.mixup > 0:
            lam = float(np.random.beta(a.mixup, a.mixup)); idx = torch.randperm(x.size(0), device=dev)
            out = model(lam * x + (1 - lam) * x[idx]); loss = lam * crit(out, y) + (1 - lam) * crit(out, y[idx])
        else:
            out = model(x); loss = crit(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        ls += loss.item() * len(y); cor += (out.argmax(1) == y).sum().item(); tot += len(y)
    return ls / tot, cor / tot


@torch.no_grad()
def evaluate(loader, tta=False):
    model.eval(); tot = cor = 0; ls = 0.0; ys, ps = [], []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev); logits = model(x)
        if tta: logits = (logits.softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)).log()
        ls += val_crit(logits, y).item() * len(y) if not tta else 0.0
        pred = logits.argmax(1); cor += (pred == y).sum().item(); tot += len(y)
        ys += y.cpu().tolist(); ps += pred.cpu().tolist()
    return ls / tot, cor / tot, ys, ps


hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
best, best_vl, bad = -1.0, 9e9, 0; ckpt = RUN / "best.pt"
for ep in range(1, a.epochs + 1):
    t0 = time.time(); tl, ta = train_epoch(); vl, vacc, _, _ = evaluate(va)
    sched.step(vacc) if plateau else sched.step()
    for k, v in zip(hist, (tl, vl, ta, vacc)): hist[k].append(v)
    print(f"Epoch {ep:02d} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {vacc:.4f} | {time.time()-t0:.0f}s")
    if vacc > best or (vacc == best and vl < best_vl):
        best, best_vl, bad = vacc, vl, 0
        torch.save({"state_dict": model.state_dict(), "class_names": names, "val_acc": vacc, "val_loss": vl, "epoch": ep,
                    "img_size": a.img_size, "seed": a.seed, "arch": a.arch, "dropout": a.dropout, "eval_mode": a.eval_mode, "args": vars(a)}, ckpt)
    else:
        bad += 1
        if bad >= a.patience: print("Early stopping."); break

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
plt.tight_layout(); plt.savefig(RUN / "curves.png", dpi=120); plt.close()
json.dump(hist, open(RUN / "history.json", "w"), indent=2)
ck = torch.load(ckpt, map_location=dev)
summary = {"run": a.run_name, "args": vars(a), "best_epoch": ck["epoch"], "best_val_acc": ck["val_acc"], "best_val_loss": ck["val_loss"]}
print(f"\nBest VAL acc {ck['val_acc']:.4f} (epoch {ck['epoch']}) -> {ckpt}")

if a.skip_test:
    json.dump(summary, open(RUN / "val_summary.json", "w"), indent=2); print("Test evaluation skipped (--skip-test)."); sys.exit()

model.load_state_dict(ck["state_dict"])
_, acc, ys, ps = evaluate(te, tta=a.tta)
p, r, f1, _ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
labels = list(range(len(names)))
metrics = {**summary, "test_accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1,
           "num_test_images": len(ys), "tta_hflip": a.tta, "device": str(dev)}
x1 = next(iter(te))[0][:1].to(dev); model.eval()
with torch.no_grad():
    for _ in range(20): model(x1)
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): model(x1)
    if dev.type == "cuda": torch.cuda.synchronize()
metrics["inference_ms_per_image_model_only"] = (time.perf_counter() - t0) / 100 * 1000
metrics["inference_timing_note"] = "batch=1, preprocessed tensor, 20 warm-up + 100 timed runs, excludes image loading and TTA"
json.dump(metrics, open(RUN / "test_metrics.json", "w"), indent=2)
(RUN / "classification_report.txt").write_text(classification_report(ys, ps, labels=labels, target_names=names, zero_division=0))
cm = confusion_matrix(ys, ps, labels=labels)
plt.figure(figsize=(18, 15)); sns.heatmap(cm, cmap="Blues", xticklabels=names, yticklabels=names, cbar=False)
plt.xticks(fontsize=5, rotation=90); plt.yticks(fontsize=5); plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.savefig(RUN / "confusion_matrix.png", dpi=130); plt.close()
if a.promote:
    shutil.copy(ckpt, C.CLASSIFIER_CKPT)
    for f in ("test_metrics.json", "classification_report.txt", "curves.png", "confusion_matrix.png"):
        shutil.copy(RUN / f, C.REPORTS / f"classifier_{f}" if f.endswith(("json", "txt")) else C.FIGURES / f"classifier_{f}")
    print("Promoted to", C.CLASSIFIER_CKPT)
print(json.dumps(metrics, indent=2))
