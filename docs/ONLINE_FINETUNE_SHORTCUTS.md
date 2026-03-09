# Online Fine-Tuning Shortcuts (Research Notes)

This document captures practical shortcuts from public docs/blogs that are most
relevant to your XFCE GUI adaptation goal.

## Key takeaway

For your current failure (missing filename input + save button), start with
detector fine-tuning (YOLO) and treat Florence-2 fine-tuning as optional phase 2.

## Sources reviewed

- OmniParser GitHub README:
  - https://github.com/microsoft/OmniParser
- Hugging Face Florence-2 fine-tuning blog:
  - https://huggingface.co/blog/finetune-florence2
- Ultralytics training best practices:
  - https://docs.ultralytics.com/guides/model-training-tips/

## Shortcut recommendations

1. YOLO-first adaptation (highest ROI)
   - Keep pretrained detector weights.
   - Fine-tune only on XFCE UI classes you care about.
   - Use transfer learning and early stopping.
   - This directly targets "element not detected" failure.

2. Human-in-the-loop label correction via rendered overlays
   - Auto-label quickly with VLM.
   - Review labels visually with box overlays.
   - Fix JSON, regenerate overlays, then export YOLO format.
   - This is faster than fully manual annotation.

3. Florence-2 QLoRA only if needed
   - Use PEFT/LoRA adapter training with 4-bit loading where possible.
   - Start with low LR (around 1e-6 to low e-5 range depending on setup).
   - Consider freezing vision tower at first for lower VRAM.
   - Trigger this step only if detector catches elements but captions still
     confuse action selection.

4. Small-data tactics for first 10-100 shots
   - Keep class taxonomy narrow and consistent.
   - Add hard negatives (screens without target controls) to reduce false
     positives.
   - Split by scene/session (not random adjacent frames only) to avoid leakage.

## Decision rule for your pipeline

- If target element is not boxed: improve YOLO data/training.
- If target element is boxed but action choice is wrong: improve captioning and
  planner prompting; then consider Florence-2 adapter.
