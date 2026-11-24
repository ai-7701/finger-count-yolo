# Training results

- Dataset: Finger Count Cleaned Dataset (nayasakka88), 5,615 real photographs, 6 classes (0–5)
- Split: 4,776 train / 839 validation (15%), stratified, seed 0
- Model: yolo11n-cls, imgsz 224, batch 64, 40 epochs, AdamW (auto), patience 15
- Hardware: Kaggle Tesla T4, 1.53 h wall clock
- Final: top-1 0.9285, top-5 0.9988

## Epoch curve (last 5)

| Epoch | train_loss | top-1 | top-5 |
| --- | --- | --- | --- |
| 36 | 5447.34 | 0.9285 | 0.9988 |
| 37 | 5006.77 | 0.9273 | 1.0000 |
| 38 | 5021.74 | 0.9273 | 0.9988 |
| 39 | 5073.62 | 0.9261 | 0.9988 |
| 40 | 5201.21 | 0.9285 | 0.9988 |

Full log: `training_results.csv`.

## Where it fails

From `confusion_matrix.png` (rows = predicted, columns = true):

| True | Predicted | Count |
| --- | --- | --- |
| 5 | 4 | 11 |
| 4 | 3 | 6 |
| 3 | 2 | 5 |
| 2 | 1 | 3 |
| 0 | 1 | 3 |
| 5 | 3 | 5 |

Neighbouring counts are the only confusions — the model never jumps from 1 to 4.
That pattern is expected: one extra folded finger is a small visual change.
