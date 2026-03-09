#!/usr/bin/env python3
"""
Web-based bounding-box review and correction tool.

Serves a local Flask app where you can:
  - View each screenshot with VLM-generated boxes overlaid
  - Drag boxes to move them
  - Resize boxes via corner handles
  - Delete boxes
  - Draw new boxes
  - Change class labels via dropdown
  - Save corrected labels to data/labels/reviewed/

Usage:
    python scripts/review_labels.py [--port 5555]
    Then open http://localhost:5555 in a browser.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = ROOT / "data" / "raw" / "screenshots"
AUTO_LABELS_DIR = ROOT / "data" / "labels" / "auto"
REVIEWED_DIR = ROOT / "data" / "labels" / "reviewed"
CLASSES_FILE = ROOT / "configs" / "classes.yaml"

app = Flask(__name__)


def load_classes() -> list[str]:
    import yaml
    data = yaml.safe_load(CLASSES_FILE.read_text())
    return data.get("classes", [])


def get_image_list() -> list[str]:
    if not SCREENSHOTS_DIR.exists():
        return []
    return sorted(p.name for p in SCREENSHOTS_DIR.glob("*.png"))


def label_path_for(image_name: str) -> Path:
    stem = Path(image_name).stem
    reviewed = REVIEWED_DIR / f"{stem}.json"
    if reviewed.exists():
        return reviewed
    auto = AUTO_LABELS_DIR / f"{stem}.json"
    return auto


# ---- API routes ----

@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/images")
def api_images():
    return jsonify(get_image_list())


@app.route("/api/classes")
def api_classes():
    return jsonify(load_classes())


@app.route("/api/labels/<image_name>")
def api_get_labels(image_name: str):
    lp = label_path_for(image_name)
    if lp.exists():
        return jsonify(json.loads(lp.read_text()))
    return jsonify({"image": image_name, "width": 0, "height": 0, "objects": []})


@app.route("/api/labels/<image_name>", methods=["POST"])
def api_save_labels(image_name: str):
    data = request.get_json(force=True)
    REVIEWED_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem
    out = REVIEWED_DIR / f"{stem}.json"
    out.write_text(json.dumps(data, indent=2))
    return jsonify({"ok": True, "path": str(out)})


@app.route("/screenshots/<path:name>")
def serve_screenshot(name: str):
    return send_from_directory(str(SCREENSHOTS_DIR), name)


# ---- Inline HTML/JS for the editor ----

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Label Review Tool</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; display: flex; height: 100vh; }
#sidebar { width: 220px; background: #16213e; padding: 12px; overflow-y: auto; flex-shrink: 0; }
#sidebar h3 { margin-bottom: 8px; font-size: 14px; color: #0f3460; }
#image-list { list-style: none; }
#image-list li { padding: 6px 8px; cursor: pointer; border-radius: 4px; font-size: 13px; margin-bottom: 2px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#image-list li:hover { background: #0f3460; }
#image-list li.active { background: #e94560; color: #fff; }
#main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
#toolbar { background: #16213e; padding: 8px 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
#toolbar button { padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
#btn-save { background: #00b894; color: #fff; }
#btn-save:hover { background: #00a381; }
#btn-prev, #btn-next { background: #0f3460; color: #fff; }
#btn-prev:hover, #btn-next:hover { background: #1a4a8a; }
#btn-add { background: #e17055; color: #fff; }
#btn-delete { background: #d63031; color: #fff; }
#class-select { padding: 4px 8px; border-radius: 4px; font-size: 13px; }
#status { margin-left: auto; font-size: 12px; color: #aaa; }
#canvas-wrap { flex: 1; overflow: auto; position: relative; background: #111; display: flex;
  align-items: center; justify-content: center; }
canvas { cursor: crosshair; }
#box-info { background: #16213e; padding: 8px 16px; font-size: 12px; min-height: 32px; }
</style>
</head>
<body>
<div id="sidebar">
  <h3>Screenshots</h3>
  <ul id="image-list"></ul>
</div>
<div id="main">
  <div id="toolbar">
    <button id="btn-prev">&larr; Prev</button>
    <button id="btn-next">Next &rarr;</button>
    <select id="class-select"></select>
    <button id="btn-add">+ Draw Box</button>
    <button id="btn-delete">Delete Selected</button>
    <button id="btn-save">Save</button>
    <span id="status"></span>
  </div>
  <div id="canvas-wrap">
    <canvas id="canvas"></canvas>
  </div>
  <div id="box-info">Select a box to see details</div>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const imageList = document.getElementById('image-list');
const classSelect = document.getElementById('class-select');
const boxInfo = document.getElementById('box-info');
const status = document.getElementById('status');

let images = [];
let classes = [];
let currentIdx = -1;
let currentImage = null;
let labelData = null;
let selectedBox = -1;
let drawMode = false;

let drag = null;  // {type: 'move'|'resize', boxIdx, corner, startX, startY, origBox}
let drawStart = null;

const HANDLE = 7;
const COLORS = [
  '#e94560','#00b894','#0984e3','#fdcb6e','#e17055',
  '#6c5ce7','#00cec9','#fab1a0','#74b9ff','#a29bfe',
  '#55efc4','#fd79a8',
];

function colorFor(cls) {
  let i = classes.indexOf(cls);
  if (i < 0) i = 0;
  return COLORS[i % COLORS.length];
}

async function init() {
  const [imgs, cls] = await Promise.all([
    fetch('/api/images').then(r => r.json()),
    fetch('/api/classes').then(r => r.json()),
  ]);
  images = imgs;
  classes = cls;
  classSelect.innerHTML = classes.map(c => `<option value="${c}">${c}</option>`).join('');
  renderImageList();
  if (images.length > 0) loadImage(0);
}

function renderImageList() {
  imageList.innerHTML = images.map((name, i) =>
    `<li data-idx="${i}" class="${i === currentIdx ? 'active' : ''}">${name}</li>`
  ).join('');
  imageList.querySelectorAll('li').forEach(li => {
    li.onclick = () => loadImage(parseInt(li.dataset.idx));
  });
}

async function loadImage(idx) {
  currentIdx = idx;
  selectedBox = -1;
  drawMode = false;
  renderImageList();
  const name = images[idx];
  status.textContent = `Loading ${name}...`;

  const img = new Image();
  img.onload = async () => {
    currentImage = img;
    canvas.width = img.width;
    canvas.height = img.height;
    const data = await fetch(`/api/labels/${name}`).then(r => r.json());
    labelData = data;
    if (!labelData.width) labelData.width = img.width;
    if (!labelData.height) labelData.height = img.height;
    if (!labelData.objects) labelData.objects = [];
    draw();
    status.textContent = `${name} — ${labelData.objects.length} boxes`;
  };
  img.src = `/screenshots/${name}`;
}

function draw() {
  if (!currentImage) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImage, 0, 0);

  (labelData.objects || []).forEach((obj, i) => {
    const [x1, y1, x2, y2] = obj.bbox_xyxy;
    const w = x2 - x1, h = y2 - y1;
    const color = colorFor(obj.class);
    const isSel = i === selectedBox;

    ctx.strokeStyle = color;
    ctx.lineWidth = isSel ? 3 : 2;
    ctx.strokeRect(x1, y1, w, h);

    // label background
    const label = `${obj.class}${obj.text ? ': ' + obj.text.slice(0, 20) : ''}`;
    ctx.font = '12px system-ui';
    const tm = ctx.measureText(label);
    const lh = 16;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(x1, y1 - lh - 2, tm.width + 8, lh + 2);
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x1 + 4, y1 - 4);

    if (isSel) {
      // draw resize handles
      ctx.fillStyle = '#fff';
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      for (const [hx, hy] of corners(obj)) {
        ctx.fillRect(hx - HANDLE/2, hy - HANDLE/2, HANDLE, HANDLE);
        ctx.strokeRect(hx - HANDLE/2, hy - HANDLE/2, HANDLE, HANDLE);
      }
    }
  });

  if (selectedBox >= 0 && selectedBox < (labelData.objects || []).length) {
    const obj = labelData.objects[selectedBox];
    boxInfo.textContent = `[${selectedBox}] ${obj.class} bbox=${JSON.stringify(obj.bbox_xyxy)} text="${obj.text || ''}"`;
  } else {
    boxInfo.textContent = 'Select a box to see details';
  }
}

function corners(obj) {
  const [x1, y1, x2, y2] = obj.bbox_xyxy;
  return [[x1, y1], [x2, y1], [x1, y2], [x2, y2]];
}

function hitCorner(obj, mx, my) {
  for (const [i, [hx, hy]] of corners(obj).entries()) {
    if (Math.abs(mx - hx) <= HANDLE && Math.abs(my - hy) <= HANDLE) return i;
  }
  return -1;
}

function hitBox(mx, my) {
  // reverse so topmost drawn box wins
  for (let i = (labelData.objects || []).length - 1; i >= 0; i--) {
    const [x1, y1, x2, y2] = labelData.objects[i].bbox_xyxy;
    if (mx >= x1 && mx <= x2 && my >= y1 && my <= y2) return i;
  }
  return -1;
}

canvas.addEventListener('mousedown', e => {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
  const mx = (e.clientX - rect.left) * sx, my = (e.clientY - rect.top) * sy;

  if (drawMode) {
    drawStart = {x: mx, y: my};
    return;
  }

  // check resize handles on selected box first
  if (selectedBox >= 0 && selectedBox < (labelData.objects||[]).length) {
    const corner = hitCorner(labelData.objects[selectedBox], mx, my);
    if (corner >= 0) {
      drag = {type: 'resize', boxIdx: selectedBox, corner,
              startX: mx, startY: my,
              origBox: [...labelData.objects[selectedBox].bbox_xyxy]};
      return;
    }
  }

  const hit = hitBox(mx, my);
  if (hit >= 0) {
    selectedBox = hit;
    classSelect.value = labelData.objects[hit].class;
    drag = {type: 'move', boxIdx: hit,
            startX: mx, startY: my,
            origBox: [...labelData.objects[hit].bbox_xyxy]};
    draw();
    return;
  }

  selectedBox = -1;
  draw();
});

canvas.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
  const mx = (e.clientX - rect.left) * sx, my = (e.clientY - rect.top) * sy;

  if (drawMode && drawStart) {
    draw();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(drawStart.x, drawStart.y, mx - drawStart.x, my - drawStart.y);
    ctx.setLineDash([]);
    return;
  }

  if (!drag) return;
  const obj = labelData.objects[drag.boxIdx];
  const dx = mx - drag.startX, dy = my - drag.startY;

  if (drag.type === 'move') {
    const [ox1, oy1, ox2, oy2] = drag.origBox;
    obj.bbox_xyxy = [
      Math.round(ox1 + dx), Math.round(oy1 + dy),
      Math.round(ox2 + dx), Math.round(oy2 + dy),
    ];
  } else if (drag.type === 'resize') {
    const b = [...drag.origBox];
    // corners: 0=TL, 1=TR, 2=BL, 3=BR
    if (drag.corner === 0) { b[0] += dx; b[1] += dy; }
    else if (drag.corner === 1) { b[2] += dx; b[1] += dy; }
    else if (drag.corner === 2) { b[0] += dx; b[3] += dy; }
    else if (drag.corner === 3) { b[2] += dx; b[3] += dy; }
    obj.bbox_xyxy = b.map(v => Math.round(v));
  }
  draw();
});

canvas.addEventListener('mouseup', e => {
  if (drawMode && drawStart) {
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * sx, my = (e.clientY - rect.top) * sy;
    const x1 = Math.round(Math.min(drawStart.x, mx));
    const y1 = Math.round(Math.min(drawStart.y, my));
    const x2 = Math.round(Math.max(drawStart.x, mx));
    const y2 = Math.round(Math.max(drawStart.y, my));
    if (x2 - x1 > 5 && y2 - y1 > 5) {
      const cls = classSelect.value || classes[0];
      labelData.objects.push({
        id: `obj-${labelData.objects.length + 1}`,
        class: cls,
        bbox_xyxy: [x1, y1, x2, y2],
        text: '',
        source: 'human',
      });
      selectedBox = labelData.objects.length - 1;
    }
    drawStart = null;
    drawMode = false;
    document.getElementById('btn-add').style.background = '#e17055';
    draw();
    return;
  }
  if (drag) {
    // normalize so x1<x2, y1<y2
    const obj = labelData.objects[drag.boxIdx];
    let [x1, y1, x2, y2] = obj.bbox_xyxy;
    obj.bbox_xyxy = [Math.min(x1,x2), Math.min(y1,y2), Math.max(x1,x2), Math.max(y1,y2)];
    drag = null;
    draw();
  }
});

// toolbar buttons
document.getElementById('btn-add').onclick = () => {
  drawMode = !drawMode;
  document.getElementById('btn-add').style.background = drawMode ? '#d63031' : '#e17055';
  document.getElementById('btn-add').textContent = drawMode ? 'Drawing... (click+drag)' : '+ Draw Box';
};

document.getElementById('btn-delete').onclick = () => {
  if (selectedBox >= 0 && labelData.objects) {
    labelData.objects.splice(selectedBox, 1);
    selectedBox = -1;
    draw();
  }
};

document.getElementById('btn-save').onclick = async () => {
  if (!labelData) return;
  const name = images[currentIdx];
  const resp = await fetch(`/api/labels/${name}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(labelData),
  });
  const result = await resp.json();
  status.textContent = result.ok ? `Saved ${name}` : 'Save failed';
};

document.getElementById('btn-prev').onclick = () => {
  if (currentIdx > 0) loadImage(currentIdx - 1);
};
document.getElementById('btn-next').onclick = () => {
  if (currentIdx < images.length - 1) loadImage(currentIdx + 1);
};

classSelect.onchange = () => {
  if (selectedBox >= 0 && labelData.objects[selectedBox]) {
    labelData.objects[selectedBox].class = classSelect.value;
    draw();
  }
};

// keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft' || e.key === '[') document.getElementById('btn-prev').click();
  else if (e.key === 'ArrowRight' || e.key === ']') document.getElementById('btn-next').click();
  else if (e.key === 'Delete' || e.key === 'Backspace') document.getElementById('btn-delete').click();
  else if (e.key === 's' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); document.getElementById('btn-save').click(); }
  else if (e.key === 'n') document.getElementById('btn-add').click();
});

init();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch label review web UI.")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--screenshots-dir", type=Path, default=None)
    parser.add_argument("--labels-dir", type=Path, default=None)
    args = parser.parse_args()

    global SCREENSHOTS_DIR, AUTO_LABELS_DIR
    if args.screenshots_dir:
        SCREENSHOTS_DIR = args.screenshots_dir
    if args.labels_dir:
        AUTO_LABELS_DIR = args.labels_dir

    REVIEWED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[review] Screenshots: {SCREENSHOTS_DIR}")
    print(f"[review] Auto labels: {AUTO_LABELS_DIR}")
    print(f"[review] Reviewed:    {REVIEWED_DIR}")
    print(f"[review] Open http://localhost:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
