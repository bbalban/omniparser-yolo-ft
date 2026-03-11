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
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"VLM output does not contain JSON object: {text[:200]}")
    raw = match.group(0)
    # Try to fix trailing commas before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to repair truncated JSON by closing open brackets
    repaired = raw.rstrip().rstrip(",")
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    # Remove any trailing incomplete object (no closing brace)
    if open_braces > 0:
        last_complete = repaired.rfind("}")
        if last_complete > 0:
            repaired = repaired[:last_complete + 1]
            open_braces = repaired.count("{") - repaired.count("}")
            open_brackets = repaired.count("[") - repaired.count("]")
    repaired = repaired.rstrip().rstrip(",")
    repaired += "]" * open_brackets + "}" * open_braces
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Last resort: extract individual complete object dicts
    obj_pattern = re.finditer(r'\{[^{}]*"class"[^{}]*"bbox_xyxy"[^{}]*\}', raw, re.DOTALL)
    objects = []
    for m in obj_pattern:
        try:
            objects.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    if objects:
        return {"objects": objects}

    raise ValueError(f"Could not parse VLM JSON: {raw[:300]}")


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
    parser.add_argument("--skip", type=str, nargs="*", default=[], help="Filenames to skip.")
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

    skip_set = set(args.skip)
    images = [p for p in images if p.name not in skip_set]

    for image_path in images:
        with Image.open(image_path) as im:
            width, height = im.size
        data_url = image_to_data_url(image_path)
        prompt = f"""\
You are a precise UI element annotator for YOLO training data. Detect all INTERACTIVE elements in this {width}x{height} pixel XFCE desktop screenshot.

Every detected element gets the same class: "icon"

Return ONLY valid JSON (no markdown fences, no explanation):
{{
  "objects": [
    {{
      "id": "obj-1",
      "class": "icon",
      "bbox_xyxy": [x1, y1, x2, y2],
      "text": "visible text on the element or empty"
    }}
  ]
}}

What to detect (all as class "icon"):
- Buttons (Save, Cancel, OK, Close, any clickable button)
- Text input fields, filename inputs, search boxes, address bars
- Menu bar items (File, Edit, View, Help)
- Dropdown menu entries
- Toolbar buttons (back, forward, up, home, refresh, icons in toolbars)
- Tab headers
- Checkboxes, radio buttons, toggle switches
- Sidebar navigation items (Home, Desktop, Documents, etc.)
- Window control buttons (close X, minimize, maximize)
- Desktop shortcut icons
- Scrollbar arrows and thumbs
- Any other clickable or typeable UI element

What NOT to detect:
- Static text, paragraphs, or labels that are not clickable
- Window borders/frames
- Status bar text
- Background, wallpaper, shadows
- File content displayed inside editors

CRITICAL rules for accuracy:
- bbox_xyxy = [left, top, right, bottom] in EXACT integer pixel coordinates.
- The bounding box must TIGHTLY wrap each element's visible border. No padding.
- Only label elements from the FOREGROUND window (topmost/active). Ignore background windows.
- Image is {width}x{height} pixels. All coordinates must be 0 <= x < {width}, 0 <= y < {height}.
- No trailing commas. Return valid JSON only."""

        parsed = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                    max_tokens=8192,
                )
                raw_text = resp.choices[0].message.content
                parsed = parse_json(raw_text)
                break
            except Exception as e:
                print(f"  [label] attempt {attempt+1}/3 failed for {image_path.name}: {e}")
                if attempt == 2:
                    print(f"  [label] SKIPPING {image_path.name}")
        if parsed is None:
            continue
        objects = parsed.get("objects", [])
        cleaned = []
        target_class = classes[0]  # single-class: always use first class
        for idx, obj in enumerate(objects, start=1):
            bbox = obj.get("bbox_xyxy", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            cleaned.append(
                {
                    "id": str(obj.get("id", f"obj-{idx}")),
                    "class": target_class,
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
