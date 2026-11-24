"""Run the trained classifier on an image or folder.

    python src/predict.py --source hand.jpg
    python src/predict.py --source photos/ --model src/finger_count_best.pt
"""
import argparse
import os
import sys

import torch
torch.backends.mkldnn.enabled = False

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="src/finger_count_best.pt")
    ap.add_argument("--source", required=True, help="image, folder, or 0 for webcam")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--save-dir", default="predictions")
    a = ap.parse_args()

    if not os.path.exists(a.model):
        sys.exit(f"model not found: {a.model}")

    model = YOLO(a.model)
    results = model.predict(source=a.source, imgsz=a.imgsz, stream=True, verbose=False)

    os.makedirs(a.save_dir, exist_ok=True)
    from PIL import Image
    for r in results:
        count = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        name = os.path.basename(r.path)
        print(f"{name}: {count} finger{'s' if count != 1 else ''} (conf {conf:.2f})")
        Image.fromarray(r.plot()[:, :, ::-1]).save(os.path.join(a.save_dir, name))


if __name__ == "__main__":
    main()
