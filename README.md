# omniparser-finetune

Practical pipeline to improve OmniParser on XFCE desktop UI elements:

1. Capture desktop screenshots for target tasks.
2. Auto-label with a VLM into bounding-box JSON.
3. Render overlays so a human can quickly fix boxes/classes.
4. Export training data for YOLO icon detection.
5. Fine-tune Florence-2 with QLoRA on icon/text descriptions.
6. Package artifacts to deploy into the Retina VM.

This repository is intentionally self-contained and only modifies files under
`omniparser-finetune/`.

## What is included

- `docs/PLAN.md`: step-by-step execution plan for 10-shot bootstrap.
- `docs/DATA_FORMAT.md`: dataset schema and class taxonomy.
- `scripts/download_upstreams.sh`: fetch upstream open-source repos/tools.
- `scripts/capture_xfce_screenshots.py`: automated screenshot capture loop.
- `scripts/label_with_vlm.py`: auto-label screenshots via OpenAI-compatible VLM.
- `scripts/render_overlays.py`: render labeled boxes on top of screenshots.
- `scripts/export_yolo_dataset.py`: convert reviewed labels to YOLO format.
- `scripts/train_yolo.py`: YOLO fine-tuning entrypoint.
- `scripts/train_florence2_qlora.py`: Florence-2 QLoRA fine-tuning entrypoint.
- `scripts/package_retina_artifacts.py`: package trained weights for Retina VM.
- `Makefile`: one-command helpers for bootstrap and 10-shot runs.

## Quick start (10-shot)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 0) Pull upstream repos/tools
make download-tools

# 1) Capture 10 screenshots (run inside XFCE session)
make capture-10

# 2) Auto-label with VLM (requires OPENAI_API_KEY and OPENAI_MODEL)
make autolabel-10

# 3) Render overlays and manually fix JSON labels as needed
make overlays

# 4) Export YOLO dataset
make export-yolo
```

Then run training on a GPU machine/VM:

```bash
# YOLO
python scripts/train_yolo.py --data data/final/yolo/dataset.yaml --epochs 40 --imgsz 1280

# Florence-2 QLoRA
python scripts/train_florence2_qlora.py \
  --train-jsonl data/final/florence/train.jsonl \
  --output-dir artifacts/florence2-qlora
```

## Notes on VM integration

- You can reuse the outer project's Lambda scripts from `../scripts/`.
- This repo produces artifacts in `artifacts/` that can be copied to Retina VM.
- Deployment/integration into Retina inference code can be done by your other
  agent once weights are validated.
