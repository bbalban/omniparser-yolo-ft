# omniparser-finetune

Practical pipeline to improve OmniParser on XFCE desktop UI elements:

1. **Capture** desktop screenshots automatically (launches XFCE apps + dialogs).
2. **Auto-label** with a VLM into bounding-box JSON.
3. **Review & fix** labels in a visual web UI (drag/resize/delete/add boxes).
4. **Export** training data for YOLO icon detection + Florence-2.
5. **Train** on a Lambda GPU VM (YOLO transfer learning, optional Florence-2 QLoRA).
6. **Deploy** artifacts to the Retina VM.

## What is included

| File | Purpose |
|------|---------|
| `PLAN.md` | Step-by-step execution plan |
| `DATA_FORMAT.md` | Label JSON schema and class taxonomy |
| `configs/classes.yaml` | XFCE UI element class list |
| `scripts/capture_xfce_screenshots.py` | Automated XFCE app launch + screenshot capture |
| `scripts/label_with_vlm.py` | Auto-label screenshots via OpenAI-compatible VLM |
| `scripts/review_labels.py` | Web UI for visual bounding-box review and correction |
| `scripts/export_yolo_dataset.py` | Convert reviewed labels to YOLO + Florence formats |
| `scripts/download_upstreams.sh` | Fetch upstream repos (OmniParser, ultralytics, etc.) |
| `requirements.txt` | Python dependencies |

## Quick start (10-shot)

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_upstreams.sh

# 1) Capture 10 screenshots (run inside XFCE session with $DISPLAY set)
python scripts/capture_xfce_screenshots.py --count 10

# 2) Auto-label with VLM
export OPENAI_API_KEY=sk-...
python scripts/label_with_vlm.py

# 3) Review & fix labels in browser
python scripts/review_labels.py --port 5555
#   → open http://localhost:5555
#   → drag/resize/delete/add boxes, change classes, Save each image

# 4) Export YOLO dataset
python scripts/export_yolo_dataset.py
```

## Training (on GPU VM)

```bash
# YOLO fine-tuning (transfer learning from OmniParser pretrained weights)
pip install ultralytics
python scripts/train_yolo.py --data data/final/yolo/dataset.yaml --epochs 40 --imgsz 1280

# Florence-2 QLoRA (optional, only if captions are the bottleneck)
pip install transformers peft bitsandbytes accelerate
python scripts/train_florence2_qlora.py \
  --train-jsonl data/final/florence/train.jsonl \
  --output-dir artifacts/florence2-qlora
```

## VM integration

- Reuse Lambda launch scripts from `../scripts/lambda_manager.py`.
- This repo produces artifacts in `artifacts/` to copy to Retina VM.
- Deployment into Retina inference code is handled by the other agent.
