# Finger Count

Count the number of raised fingers on one hand, in real time, **entirely in the browser**.

MediaPipe's hand landmarker finds the hand and crops it; a YOLO11n classifier trained on
5,615 real photographs counts the fingers. No server, no image upload — the model runs
on the visitor's device via ONNX Runtime Web.

**Live demo:** https://ai-7701.github.io/finger-count-yolo/

---

## Results

Measured on 839 held-out photographs never seen during training.

| Metric | Value |
| --- | --- |
| Top-1 accuracy | **92.85%** |
| Top-5 accuracy | 99.88% |
| Validation images | 839 (15% split) |
| Model | YOLO11n-cls, 1.53 M parameters, 3.2 MB |
| Training | 40 epochs, Kaggle T4 GPU, 1.5 hours |
| Inference (T4) | 1.3 ms/image |
| Export | ONNX opset 12, 5.9 MB |

Per-class behaviour from `docs/confusion_matrix.png`: the model rarely confuses two
neighbouring counts. The largest errors are 5 read as 4 (11 of 147 cases) and 4 read as 3
(6 of 142).

## How it works

```
webcam frame
   │
   ├─ MediaPipe HandLandmarker  →  21 landmarks  →  bounding box (+25% padding)
   │
   ├─ crop → resize 224×224 → float32 NCHW ÷ 255
   │
   └─ YOLO11n-cls (ONNX, WASM)  →  6 logits  →  softmax  →  count + confidence
```

The classifier is the model; MediaPipe is only there to find and crop the hand. That split
matters: the training dataset has folder labels (`0/ … 5/`) and **no bounding boxes**, so a
detector trained on it would have learned boxes we invented. Cropping with a real hand
detector keeps both the box and the count honest.

## Run it locally

```bash
cd site
python -m http.server 8000
# open http://localhost:8000
```

A local server is required — `file://` blocks the WASM and model fetches.
The runtime libraries are vendored in `site/vendor/`, so it works offline.
CDN copies are used automatically if the vendored files are missing.

Add `?autotest=1` to the URL to run a self-check without a camera: it loads the runtime,
runs one inference on a synthetic input, and prints the result in the status line.

## Repository layout

```
site/                      static web app (GitHub Pages root)
  index.html               landing page + live demo
  js/app.js                MediaPipe + ONNX Runtime pipeline
  css/style.css
  models/finger_count.onnx exported classifier (5.9 MB)
  vendor/ort/              ONNX Runtime Web + WASM
  vendor/mp/               MediaPipe tasks-vision + hand_landmarker.task
src/finger_count_best.pt   trained PyTorch weights (3.2 MB)
notebooks_train_kaggle.ipynb  full training notebook (Kaggle, T4)
docs/                      confusion matrix, training log
```

## Retrain

1. Open `notebooks_train_kaggle.ipynb` in Kaggle.
2. Add input: **Finger Count Cleaned Dataset** (nayasakka88).
3. Settings → Accelerator **GPU T4 x2**, Internet **On**.
4. Run all cells. 40 epochs takes ~1.5 hours.

The notebook builds a train/val split with symlinks, trains `yolo11n-cls`, evaluates, and
exports ONNX into `deliverables/`.

## Limitations

- **One hand only.** With two hands in frame it counts whichever it detects first.
- **Raised fingers, not gestures.** A thumbs-up reads as 1; this is not sign language.
- **Needs reasonable light.** A dim room defeats the hand detector before the classifier runs.
- **Fingers should be separated.** A tight fist or overlapping fingers can be read one count off.

## Credits and licence

- Dataset: [Finger Count Cleaned Dataset](https://www.kaggle.com/datasets/nayasakka88/finger-count-cleaned-dataset) (CC0).
- Model: YOLO11 via [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- Hand detection: [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Apache-2.0).
- Site code: MIT (see `LICENSE`). Note that the AGPL-3.0 weights carry their own terms.

