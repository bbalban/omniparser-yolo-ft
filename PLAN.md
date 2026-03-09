# XFCE OmniParser Fine-Tuning Plan

## Goal

Improve detection of XFCE desktop elements that currently fail in your task
flow (notably filename input fields and save buttons during file save/edit).

## Recommended shortcut path (fastest likely win)

Based on current public guidance and your failure mode, the shortest path is:

1. Fine-tune detection first (YOLO) on XFCE controls.
2. Keep Florence-2 frozen initially.
3. Re-test the failing desktop save/edit task.
4. Only add Florence-2 QLoRA if grounding still fails due to poor captions.

Why this is a shortcut: your observed misses are primarily missing boxes
(filename field and save button), which is detector recall. Caption quality
matters after detection exists.

## Phase 0: Bootstrap (today)

1. Pull upstream repos (OmniParser, Ultralytics, Transformers, PEFT, TRL).
2. Install local tooling (`requirements.txt`).
3. Define minimal class taxonomy in `configs/classes.yaml`.

Deliverable: runnable data pipeline in this repo.

## Phase 1: 10-shot data loop

```bash
# Step 1 — Capture 10 XFCE screenshots (run inside XFCE X session)
python scripts/capture_xfce_screenshots.py --count 10

# Step 2 — Auto-label with VLM (requires OPENAI_API_KEY)
python scripts/label_with_vlm.py

# Step 3 — Review & fix labels in browser (drag/resize/delete/add boxes)
python scripts/review_labels.py --port 5555
#   → open http://localhost:5555, fix boxes, click Save per image

# Step 4 — Export to YOLO + Florence format
python scripts/export_yolo_dataset.py
```

The capture script launches 10 XFCE scenarios automatically:
mousepad save-as, thunar browsing, context menus, settings panels,
terminal, edit menus, appearance dialog, rename dialog, find/replace,
and clean desktop.

The review tool is a web UI where you visually drag/resize/delete/add
bounding boxes and change class labels — no JSON editing needed.

Deliverables:
- `data/raw/screenshots/*.png`
- `data/labels/reviewed/*.json` (human-corrected)
- `data/final/yolo/` (YOLO dataset + dataset.yaml)
- `data/final/florence/train.jsonl`

## Phase 2: Training on Retina GPU VM

1. Launch Lambda GPU VM using outer scripts (`../scripts/lambda_manager.py`).
2. Copy this repo to VM (rsync/scp).
3. Train YOLO:
   - `python scripts/train_yolo.py --data data/final/yolo/dataset.yaml`
4. Re-run evaluation on failing save/edit trajectory.
5. If needed, prepare Florence samples:
   - `scripts/export_yolo_dataset.py` also writes Florence jsonl.
6. Train Florence-2 with QLoRA:
   - `python scripts/train_florence2_qlora.py --train-jsonl ...`

Deliverables:
- `artifacts/yolo/weights/best.pt`
- `artifacts/florence2-qlora/` (adapter)

## Phase 3: Validation and integration

1. Run held-out task traces for "edit + save file on desktop".
2. Compare old vs tuned model detections:
   - filename field recall
   - save-button recall
3. Package artifacts: `scripts/package_retina_artifacts.py`.
4. Hand off artifacts + config mapping to Retina integration agent.

## Human-in-the-loop review workflow

The review tool (`scripts/review_labels.py`) serves a web UI at
http://localhost:5555 with these features:

- **Move boxes**: click and drag any box
- **Resize boxes**: drag corner handles on selected box
- **Delete boxes**: select a box and press Delete or click Delete button
- **Draw new boxes**: click "+ Draw Box", then click-drag on the image
- **Change class**: select a box, pick new class from dropdown
- **Save**: click Save (or Ctrl+S) to write corrected JSON
- **Navigate**: arrow keys or Prev/Next buttons

Keyboard shortcuts: `[`/`]` or arrows = prev/next, `n` = draw mode,
`Delete` = remove selected, `Ctrl+S` = save.

If volume grows past ~50 images, migrate to Label Studio or CVAT.

## Scale-up path after 10-shot

1. Increase to 100-300 screenshots over diverse XFCE states:
   - file manager, text editor, dialog windows, context menus, settings.
2. Keep class list stable to avoid label drift.
3. Run active-learning cycle:
   - inference on new screenshots
   - review low-confidence outputs first
   - retrain incrementally.

## Online-backed methods used in this plan

1. OmniParser architecture separates detection and captioning models, enabling
   independent adaptation.
2. Ultralytics transfer-learning guidance recommends pretrained weights +
   small-dataset adaptation, which fits the 10-shot bootstrap.
3. Florence-2 fine-tuning guidance shows low learning rate and optional frozen
   vision tower as a resource-efficient baseline.
