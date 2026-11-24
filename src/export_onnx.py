"""Export the trained classifier to ONNX for the browser demo.

    python src/export_onnx.py --model src/finger_count_best.pt --out site/models/finger_count.onnx
"""
import argparse
import shutil
import os

import torch
torch.backends.mkldnn.enabled = False

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="src/finger_count_best.pt")
    ap.add_argument("--out", default="site/models/finger_count.onnx")
    ap.add_argument("--imgsz", type=int, default=224)
    a = ap.parse_args()

    model = YOLO(a.model)
    produced = model.export(format="onnx", imgsz=a.imgsz, opset=12)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    shutil.copy(produced, a.out)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
