# YOLO Fine-Tuning Plan for XFCE Desktop UI

## Overview

Fine-tune OmniParser's YOLOv8 icon detector to reliably detect XFCE desktop
UI elements (buttons, inputs, menus, toolbar icons) that the vanilla model misses.

**Dataset:** 22 images, ~1440 bounding boxes, single class (`icon`).
**Method:** Transfer learning from OmniParser's pre-trained `icon_detect/model.pt`.

---

## Step 1: Launch GPU VM

```bash
cd ~/ai-project
python scripts/lambda_manager.py launch --gpu-count 1 --instance-type gpu_1x_a100
```

Once the VM is up, rsync the project and dataset:

```bash
rsync -avz --exclude '.git' --exclude 'third_party' \
  ~/ai-project/omniparser-finetune/ ubuntu@<RETINA_IP>:~/omniparser-finetune/
```

## Step 2: Train on GPU VM

SSH into the GPU VM and run:

```bash
cd ~/omniparser-finetune
pip install ultralytics
python scripts/train_yolo.py
```

This runs YOLOv8 fine-tuning using OmniParser weights as the starting point.
Training takes ~10-15 minutes on an A100 for 100 epochs on this dataset size.

**What to watch:**
- `box_loss` should decrease
- `mAP50` should climb toward 0.8+
- Output weights land in `runs/detect/xfce_finetune/weights/best.pt`

## Step 3: Deploy fine-tuned weights

Copy the trained weights back and replace the Retina VM model:

```bash
# From client VM:
scp ubuntu@<RETINA_IP>:~/omniparser-finetune/runs/detect/xfce_finetune/weights/best.pt \
  ~/ai-project/weights/icon_detect/model.pt
```

Restart the Retina API service so it loads the new weights.

## Step 4: Test from client VM

Run the test script from this project directory on the client VM.
It sends screenshots to the Retina VM's `/analyze` API and compares
detections before/after fine-tuning:

```bash
cd ~/ai-project/omniparser-finetune
python scripts/test_finetuned_model.py --retina-url http://<RETINA_IP>:8000
```

This will:
1. Send several XFCE screenshots to the Retina API
2. Print detected elements with confidence scores
3. Render overlay images showing what the model found
4. Specifically check if save buttons and filename inputs are detected

## How to know it worked

- The model detects XFCE save buttons, filename inputs, and menu items
  that the vanilla model missed.
- `mAP50` on the validation set is above 0.7 (ideally 0.85+).
- On new unseen XFCE screenshots, bounding boxes appear on interactive
  elements with reasonable confidence (>0.3).
- The main project's smoketest (`test_text_editor_xfce.py`) passes
  the file-save task that previously failed.
