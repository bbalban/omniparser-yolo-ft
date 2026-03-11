#!/usr/bin/env python3
"""
Fine-tune YOLOv8 on XFCE desktop UI elements.

Requires: pip install ultralytics

Usage (on GPU VM):
    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 200 --base-weights /path/to/model.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "final" / "yolo" / "dataset.yaml"
DEFAULT_BASE_WEIGHTS = ROOT / "weights" / "icon_detect" / "model.pt"
YOLOV8S_PRETRAINED = "yolov8s.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for XFCE UI detection.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-weights", type=str, default=str(DEFAULT_BASE_WEIGHTS),
                        help="Starting weights. Use OmniParser icon_detect/model.pt for transfer learning, "
                             "or 'yolov8s.pt' for a generic pretrained start.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=str, default=str(ROOT / "runs" / "detect"))
    parser.add_argument("--name", type=str, default="xfce_finetune")
    parser.add_argument("--device", type=str, default="0", help="CUDA device, e.g. '0' or 'cpu'")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"[train] Dataset not found: {args.dataset}")
        print("[train] Run scripts/export_yolo_dataset.py first.")
        raise SystemExit(1)

    weights_path = Path(args.base_weights)
    if not weights_path.exists():
        print(f"[train] Base weights not found at {weights_path}")
        print(f"[train] Falling back to pretrained {YOLOV8S_PRETRAINED}")
        args.base_weights = YOLOV8S_PRETRAINED

    from ultralytics import YOLO

    print(f"[train] Loading base model: {args.base_weights}")
    model = YOLO(args.base_weights)

    print(f"[train] Dataset: {args.dataset}")
    print(f"[train] Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}")
    print(f"[train] Output: {args.project}/{args.name}")
    print()

    results = model.train(
        data=str(args.dataset),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        exist_ok=True,
        pretrained=True,
        patience=20,
        save=True,
        save_period=25,
        plots=True,
        verbose=True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\n[train] Training complete.")
        print(f"[train] Best weights: {best_weights}")
        print(f"[train] To deploy: copy {best_weights} -> weights/icon_detect/model.pt on Retina VM")
    else:
        print("\n[train] Training finished but best.pt not found. Check logs above.")


if __name__ == "__main__":
    main()
