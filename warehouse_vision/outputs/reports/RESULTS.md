# Results (auto-generated from outputs/reports/*.json)

## Experiment 1 - ResNet-18 fine-grained classification (81 classes, official test split)

| Model | Test acc | Macro P | Macro R | Macro F1 |
|---|---|---|---|---|
| Baseline (lr 1e-3, squash eval) | 0.7594 | 0.7905 | 0.7717 | 0.7687 |
| **Final (tuned)** | **0.8773** | **0.9026** | **0.8912** | **0.8904** |

Images: 2485 | inference 2.78 ms/img (batch=1, preprocessed tensor, 20 warm-up + 100 timed runs, excludes image loading and TTA) | args: `{'lr': 0.0003, 'cosine': True, 'aug': 'strong', 'ls': 0.1, 'dropout': 0.2, 'mixup': 0.0, 'img_size': 288, 'eval_mode': 'crop', 'iconic': False, 'seed': 42}`

### Validation-only tuning (test set untouched)

| Run | Val acc | Val loss |
|---|---|---|
| t0_prev_recipe | 0.7939 | 0.7524 |
| t1_regularised | 0.8514 | 0.7782 |
| t2_plus_iconic | 0.8547 | 0.7522 |
| t3_plus_mixup | 0.8514 | 0.8869 |
| t4_img288 | 0.8851 | 0.7207 |

## Experiment 2 - YOLOv8n barcode-region detection (InventBar, single class)

imgsz 640, batch 16, seed 42, ultralytics 8.4.190

| Split | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---|---|---|---|---|
| val | 0.974 | 1.000 | 0.987 | 0.850 |
| test | 0.975 | 1.000 | 0.985 | 0.821 |

P/R at the F1-optimal confidence reported by Ultralytics; mAP at IoU 0.50 and averaged over 0.50:0.95; eval conf=0.001, NMS iou=0.6

## Barcode decoding (pyzbar, not a trained model) - decode RATE only

| Method | Images decoded |
|---|---|
| Ground-truth crop | 97.5% |
| Full image | 97.5% |
| YOLO detection crop (end-to-end) | 96.2% |

n = 80 test images; symbologies: {'EAN13': 26, 'CODE128': 27, 'CODE39': 22, 'EAN8': 3}. decode RATE only; no ground-truth text exists to verify correctness.

## Confidence policy (threshold chosen on validation)

Threshold **0.4** - smallest threshold with validation accepted-accuracy >= 0.95 and coverage >= 0.5 (target met on val: True).

| Split | Coverage | Accuracy of confident predictions | ECE |
|---|---|---|---|
| val | 77.4% | 95.6% | 0.205 |
| test | 83.3% | 93.7% | 0.163 |
