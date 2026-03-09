#!/usr/bin/env python3
"""
Export reviewed label JSONs to YOLO object-detection format.

Reads:  data/labels/reviewed/*.json  (falls back to data/labels/auto/*.json)
Writes: data/final/yolo/
          ├── dataset.yaml
          ├── images/
          │     ├── train/  (80%)
          │     └── val/    (20%)
          └── labels/
                ├── train/
                └── val/

Also writes Florence-2 training pairs to:
        data/final/florence/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = ROOT / "data" / "raw" / "screenshots"
REVIEWED_DIR = ROOT / "data" / "labels" / "reviewed"
AUTO_DIR = ROOT / "data" / "labels" / "auto"
CLASSES_FILE = ROOT / "configs" / "classes.yaml"


def load_classes(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    return data.get("classes", [])


def find_label(image_stem: str) -> Path | None:
    reviewed = REVIEWED_DIR / f"{image_stem}.json"
    if reviewed.exists():
        return reviewed
    auto = AUTO_DIR / f"{image_stem}.json"
    if auto.exists():
        return auto
    return None


def bbox_xyxy_to_yolo(bbox: list[int], img_w: int, img_h: int) -> tuple[float, ...]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return (cx, cy, w, h)


def objects_to_florence_text(objects: list[dict], classes: list[str]) -> str:
    parts = []
    for obj in objects:
        cls = obj["class"]
        text = obj.get("text", "")
        bbox = obj["bbox_xyxy"]
        desc = f"{cls}"
        if text:
            desc += f'("{text}")'
        desc += f" at [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
        parts.append(desc)
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export labels to YOLO format.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "final" / "yolo")
    parser.add_argument("--florence-out", type=Path, default=ROOT / "data" / "final" / "florence" / "train.jsonl")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    classes = load_classes(CLASSES_FILE)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    images = sorted(SCREENSHOTS_DIR.glob("*.png"))
    if not images:
        print("[export] No screenshots found.")
        return

    # Pair images with labels
    pairs = []
    for img_path in images:
        lbl_path = find_label(img_path.stem)
        if lbl_path is None:
            print(f"[export] WARNING: no label for {img_path.name}, skipping")
            continue
        pairs.append((img_path, lbl_path))

    if not pairs:
        print("[export] No image-label pairs to export.")
        return

    random.seed(args.seed)
    random.shuffle(pairs)
    split_idx = max(1, int(len(pairs) * (1 - args.val_split)))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Create output dirs
    for split in ("train", "val"):
        (args.out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    args.florence_out.parent.mkdir(parents=True, exist_ok=True)

    florence_lines = []

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for img_path, lbl_path in split_pairs:
            label_data = json.loads(lbl_path.read_text())
            img_w = label_data.get("width", 1920)
            img_h = label_data.get("height", 1080)
            objects = label_data.get("objects", [])

            # Copy image
            dst_img = args.out_dir / "images" / split_name / img_path.name
            shutil.copy2(img_path, dst_img)

            # Write YOLO label
            yolo_lines = []
            for obj in objects:
                cls = obj.get("class", "")
                if cls not in class_to_idx:
                    continue
                cx, cy, w, h = bbox_xyxy_to_yolo(obj["bbox_xyxy"], img_w, img_h)
                yolo_lines.append(f"{class_to_idx[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            lbl_out = args.out_dir / "labels" / split_name / f"{img_path.stem}.txt"
            lbl_out.write_text("\n".join(yolo_lines) + "\n" if yolo_lines else "")

            # Florence pair
            florence_lines.append(json.dumps({
                "image": str(dst_img.resolve()),
                "text": objects_to_florence_text(objects, classes),
                "split": split_name,
            }))

    # Write dataset.yaml
    dataset_yaml = {
        "path": str(args.out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    yaml_path = args.out_dir / "dataset.yaml"
    yaml_path.write_text(yaml.dump(dataset_yaml, default_flow_style=False))

    # Write Florence jsonl
    args.florence_out.write_text("\n".join(florence_lines) + "\n")

    print(f"[export] YOLO dataset: {args.out_dir}")
    print(f"[export]   train: {len(train_pairs)} images")
    print(f"[export]   val:   {len(val_pairs)} images")
    print(f"[export]   classes: {len(classes)}")
    print(f"[export] Florence JSONL: {args.florence_out} ({len(florence_lines)} pairs)")
    print(f"[export] YOLO config: {yaml_path}")


if __name__ == "__main__":
    main()
