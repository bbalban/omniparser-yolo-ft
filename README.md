# omniparser-finetune

Fine-tune OmniParser's YOLOv8 icon detector for XFCE desktop UI elements.

## Pipeline

1. **Capture** XFCE screenshots (automated + manual via `scrot`)
2. **Auto-label** with Gemini 2.5 Pro VLM (single class: `icon`)
3. **Review & fix** labels in a visual web UI (draw/drag/resize/delete boxes)
4. **Export** to YOLO training format
5. **Train** on a Lambda GPU VM (transfer learning from OmniParser weights)
6. **Test** from client VM via Retina API
7. **Deploy** fine-tuned weights to Retina VM

## Files

| File | Purpose |
|------|---------|
| `TRAINING_PLAN.md` | Step-by-step training and deployment plan |
| `PLAN.md` | Original detailed execution plan |
| `DATA_FORMAT.md` | Label JSON schema |
| `configs/classes.yaml` | Class list (single class: `icon`) |
| `scripts/capture_xfce_screenshots.py` | Automated XFCE app launch + screenshot capture |
| `scripts/label_with_vlm.py` | Auto-label via Gemini/OpenAI-compatible VLM |
| `scripts/review_labels.py` | Web UI for bounding-box review and correction |
| `scripts/export_yolo_dataset.py` | Convert reviewed labels to YOLO format |
| `scripts/train_yolo.py` | YOLOv8 fine-tuning script (run on GPU VM) |
| `scripts/test_finetuned_model.py` | Test model via Retina API or local weights |

## Quick start

```bash
pip install -r requirements.txt

# 1) Capture screenshots (inside XFCE session)
python scripts/capture_xfce_screenshots.py --count 10

# 2) Auto-label with VLM
export OPENAI_API_KEY=<your-gemini-or-openai-key>
python scripts/label_with_vlm.py --model gemini-2.5-pro \
  --base-url "https://generativelanguage.googleapis.com/v1beta/openai/"

# 3) Review labels in browser
python scripts/review_labels.py --port 5555
# Open http://localhost:5555, press N to draw, Ctrl+S to save

# 4) Export YOLO dataset
python scripts/export_yolo_dataset.py
```

## Training (on GPU VM)

```bash
pip install ultralytics
python scripts/train_yolo.py
```

See `TRAINING_PLAN.md` for full deployment steps.

## Testing (from client VM)

```bash
python scripts/test_finetuned_model.py --retina-url http://<RETINA_IP>:8000
```

## Dataset stats

- 18 reviewed images, 1304 bounding boxes
- Train/val split: 17/5 images
- Single class: `icon` (all interactive UI elements)
