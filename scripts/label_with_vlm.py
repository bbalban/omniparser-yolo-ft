#!/usr/bin/env python3
"""Auto-label screenshots with an OpenAI-compatible VLM."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI
from PIL import Image


def load_classes(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes = data.get("classes", [])
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"classes not found in {path}")
    return [str(c) for c in classes]


def image_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("VLM output does not contain JSON object.")
    return json.loads(match.group(0))


def clamp_bbox_xyxy(bbox: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label GUI screenshots with a VLM.")
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/screenshots"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/labels/auto"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.yaml"))
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4.1"))
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--limit", type=int, default=0, help="0 means all images.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    classes = load_classes(args.classes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    images = sorted(args.images_dir.glob("*.png"))
    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        print("[label] No images found.")
        return

    for image_path in images:
        with Image.open(image_path) as im:
            width, height = im.size
        data_url = image_to_data_url(image_path)
        prompt = f"""
You are labeling XFCE desktop UI elements for object detection.
Allowed classes: {classes}

Return strict JSON object with:
{{
  "objects": [
    {{
      "id": "obj-1",
      "class": "one class from allowed list",
      "bbox_xyxy": [x1, y1, x2, y2],
      "text": "visible text or empty"
    }}
  ]
}}

Constraints:
- Use integer pixel coordinates.
- Keep only visible actionable UI elements.
- Do not include objects outside image bounds ({width}x{height}).
- No markdown, no explanation, JSON only.
"""
        resp = client.responses.create(
            model=args.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )
        parsed = parse_json(resp.output_text)
        objects = parsed.get("objects", [])
        cleaned = []
        for idx, obj in enumerate(objects, start=1):
            cls = str(obj.get("class", "")).strip()
            if cls not in classes:
                continue
            bbox = obj.get("bbox_xyxy", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            cleaned.append(
                {
                    "id": str(obj.get("id", f"obj-{idx}")),
                    "class": cls,
                    "bbox_xyxy": clamp_bbox_xyxy([float(v) for v in bbox], width, height),
                    "text": str(obj.get("text", "")),
                    "source": "vlm",
                }
            )
        out = {
            "image": image_path.name,
            "width": width,
            "height": height,
            "objects": cleaned,
        }
        out_path = args.out_dir / f"{image_path.stem}.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[label] {image_path.name} -> {out_path.name} ({len(cleaned)} objects)")


if __name__ == "__main__":
    main()
