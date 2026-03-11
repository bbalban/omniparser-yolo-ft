#!/usr/bin/env python3
"""
Test fine-tuned YOLO model via the Retina VM API.

Sends XFCE screenshots to the Retina /analyze endpoint and checks whether
the fine-tuned model detects elements the vanilla model missed (save buttons,
filename inputs, menu items, etc.).

Run from the client VM:
    python scripts/test_finetuned_model.py --retina-url http://<RETINA_IP>:8000

Or test locally with a weights file (no Retina VM needed):
    python scripts/test_finetuned_model.py --local --weights runs/detect/xfce_finetune/weights/best.pt
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = ROOT / "data" / "raw" / "screenshots"
RESULTS_DIR = ROOT / "data" / "test_results"


def test_via_api(retina_url: str, images: list[Path], confidence: float) -> list[dict]:
    import requests

    results = []
    for img_path in images:
        img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        payload = {
            "image": img_b64,
            "context": "global-scan",
            "origin_coordinates": [0, 0],
            "confidence_threshold": confidence,
        }
        try:
            resp = requests.post(f"{retina_url}/analyze", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])
            results.append({
                "image": img_path.name,
                "num_elements": len(elements),
                "elements": elements,
                "processing_time_ms": data.get("processing_time_ms", 0),
            })
            print(f"  {img_path.name}: {len(elements)} elements ({data.get('processing_time_ms', 0):.0f}ms)")
            for elem in elements:
                etype = elem.get("type", "?")
                label = elem.get("label", "?")
                box = elem.get("box_2d", [])
                conf = elem.get("confidence", 0)
                print(f"    [{etype}] {label} box={box} conf={conf:.2f}")
        except Exception as e:
            print(f"  {img_path.name}: ERROR - {e}")
            results.append({"image": img_path.name, "error": str(e)})

    return results


def test_local(weights_path: str, images: list[Path], confidence: float) -> list[dict]:
    from ultralytics import YOLO
    from PIL import Image

    model = YOLO(weights_path)
    results_list = []

    for img_path in images:
        preds = model.predict(source=str(img_path), conf=confidence, imgsz=640, verbose=False)
        boxes = []
        for r in preds:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_val = float(box.conf[0])
                boxes.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf_val,
                })
        results_list.append({
            "image": img_path.name,
            "num_detections": len(boxes),
            "detections": boxes,
        })
        print(f"  {img_path.name}: {len(boxes)} detections")
        for b in boxes:
            print(f"    box={b['box']} conf={b['confidence']:.2f}")

    return results_list


def render_overlays(images: list[Path], results: list[dict]) -> None:
    from PIL import Image, ImageDraw

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    COLORS = ["red", "lime", "cyan", "magenta", "yellow", "orange"]

    for img_path, result in zip(images, results):
        im = Image.open(img_path).copy()
        draw = ImageDraw.Draw(im)

        elements = result.get("elements") or result.get("detections") or []
        for i, elem in enumerate(elements):
            box = elem.get("box_2d") or elem.get("box")
            if not box or len(box) != 4:
                continue
            color = COLORS[i % len(COLORS)]
            draw.rectangle(box, outline=color, width=2)
            label = elem.get("label", "")
            conf = elem.get("confidence", 0)
            draw.text((box[0] + 2, max(0, box[1] - 12)), f"{label} {conf:.2f}", fill=color)

        out = RESULTS_DIR / f"test_{img_path.stem}.png"
        im.save(out)
        print(f"  Overlay: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test fine-tuned YOLO model.")
    parser.add_argument("--retina-url", type=str, default="http://localhost:8000",
                        help="Retina VM API URL.")
    parser.add_argument("--local", action="store_true",
                        help="Test locally with YOLO weights instead of Retina API.")
    parser.add_argument("--weights", type=str,
                        default=str(ROOT / "runs" / "detect" / "xfce_finetune" / "weights" / "best.pt"))
    parser.add_argument("--images-dir", type=Path, default=SCREENSHOTS_DIR)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    parser.add_argument("--overlay", action="store_true", default=True,
                        help="Generate overlay images showing detections.")
    args = parser.parse_args()

    images = sorted(args.images_dir.glob("*.png"))
    images = [p for p in images if "calibration" not in p.name]
    if args.limit > 0:
        images = images[:args.limit]

    if not images:
        print("[test] No images found.")
        return

    print(f"[test] Testing {len(images)} images, confidence threshold={args.confidence}")
    print()

    if args.local:
        print(f"[test] Local mode, weights: {args.weights}")
        results = test_local(args.weights, images, args.confidence)
    else:
        print(f"[test] API mode, Retina URL: {args.retina_url}")
        results = test_via_api(args.retina_url, images, args.confidence)

    total = sum(r.get("num_elements", r.get("num_detections", 0)) for r in results)
    print(f"\n[test] Total: {total} detections across {len(images)} images")

    if args.overlay:
        print("\n[test] Generating overlay images...")
        try:
            render_overlays(images, results)
        except ImportError:
            print("  (Pillow not available, skipping overlays)")

    out_path = RESULTS_DIR / "test_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[test] Results saved: {out_path}")


if __name__ == "__main__":
    main()
