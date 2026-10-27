# Smart Warehouse Vision — Automated Product Identification and Classification in Warehouse Management

Evidence-aware, two-stage product-identification **prototype** (college mini-project). Not validated for commercial warehouse use.

## What comes from where (do not blur these in the report/demo)
| Output | Source |
|---|---|
| Fine-grained class + top-3 + confidence | **Trained model** — ResNet-18 (ImageNet-pretrained, fine-tuned on Grocery Store Dataset) |
| Barcode region boxes + detection confidence | **Trained model** — YOLOv8n (COCO-pretrained, fine-tuned on InventBar) |
| Decoded barcode value + symbology | **Deterministic library** — pyzbar/ZBar (no learning) |
| Product name / SKU / expected quantity | **Simulated local catalogue** (`data/catalogue.csv`, rows marked `SIMULATED`) |
| Verification status | **Explicit rule set** in `src/verification.py` |

The two public datasets have **no shared product identities**. A grocery class label (e.g. "Orange") is *not* a SKU/GTIN and is never used to look up the catalogue unless a catalogue row explicitly maps it (`visual_class` column, empty by default). The classification, detection and decoding experiments are separate.

## Contribution (honest framing)
Not a new algorithm. The engineering contribution is an **evidence-aware verification workflow** that keeps five kinds of evidence separate — visual prediction, barcode location, decoded value, catalogue-confirmed identity, inventory status — and handles uncertain, missing and conflicting evidence explicitly (confidence threshold chosen on validation data; "unreadable", "unknown barcode", "visual prediction only — SKU not verified", "conflicting evidence" outcomes).

## Setup (Ubuntu, AWS EC2 with NVIDIA T4)
```bash
sudo apt update && sudo apt install -y libzbar0 git unzip tmux
python3 -m venv venv && source venv/bin/activate && unset LD_LIBRARY_PATH   # system CUDA paths broke cuDNN on our instance
pip install -r requirements.txt
mkdir -p data/raw && cd data/raw
git clone https://github.com/marcusklasson/GroceryStoreDataset
git clone https://github.com/ChommaphatMa/BenchmarkBarcodeDatasets
cd ../..
```

## Reproduce the experiments
```bash
python scripts/01_verify_grocery.py          # parses manifests, checks files/labels/overlap, sample figure
python scripts/02_prepare_inventbar.py       # validates YOLO labels, seeded 70/15/15 split, data.yaml
./scripts/04_tune_classifier.sh              # validation-only tuning (test untouched)
python scripts/03_train_classifier.py --run-name final --cosine --aug strong --ls 0.1 --dropout 0.2 \
       --lr 3e-4 --epochs 30 --patience 10 --img-size 288 --promote      # test evaluated once
python scripts/05_train_detector.py          # YOLOv8n; val-selected, test evaluated once
python scripts/06_visualize_detections.py
python scripts/07_decode_study.py --add-to-catalogue 5   # decode rates; adds SIMULATED demo catalogue rows
python scripts/09_calibrate_threshold.py     # "uncertain" threshold chosen on val, effect reported on test
python scripts/08_selftest.py                # unit tests of verification rules
python scripts/10_make_results.py            # writes outputs/reports/RESULTS.md from the JSON reports
```
Dataset facts verified from the files (not assumed): Grocery manifests are comma-separated `path, fine_id, coarse_id`, paths relative to `dataset/`; 81 classes, 2,640 / 296 / 2,485 train/val/test images (5,421 total); InventBar = 527 images with native normalised YOLO boxes, one class, one box per image (no conversion needed; seeded 368/79/80 split).

## Run the app
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
On EC2, prefer an **SSH tunnel** instead of opening the port publicly: `ssh -L 8501:localhost:8501 -i key.pem ubuntu@<ip>` then open http://localhost:8501. If you open port 8501 in the security group, restrict it to **your IP**. Run inside `tmux` so it survives disconnects, and **stop the instance** when finished to avoid charges. The app still runs if only the classifier, only the detector, or neither is trained.

## Results
See `outputs/reports/RESULTS.md` (generated). Report numbers only from those files.

## Limitations to state in the report
- InventBar: one large, fairly centred barcode per image; the detector is **not** shown to work on cluttered shelves or multi-barcode scenes. Test set is 80 images, so small metric differences are noise.
- Decode figures are **decode rates** (a value was returned); InventBar has no ground-truth barcode text, so correctness is unmeasured. pyzbar checks EAN/UPC check digits, which reduces but does not eliminate wrong reads.
- Full-image decoding performed about as well as detector crops on this data; the detector's value here is localisation evidence and separating product barcodes from stray QR codes, not a higher decode rate.
- Classifier errors concentrate in visually similar produce (citrus, green/yellow apples). Softmax probability is a score, not a guarantee; see ECE in RESULTS.md.
- The catalogue is simulated. One image classification is **not** a stock count; real counting needs defined counting and duplicate-prevention logic.
- Training images come from specific stores/backgrounds; no claim is made about other environments.
