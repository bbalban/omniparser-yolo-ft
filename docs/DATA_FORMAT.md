# Data Format

## Label JSON schema

Each screenshot has one JSON file with this shape:

```json
{
  "image": "shot_0001.png",
  "width": 1920,
  "height": 1080,
  "objects": [
    {
      "id": "obj-1",
      "class": "input_filename",
      "bbox_xyxy": [100, 200, 320, 240],
      "text": "Untitled Document",
      "source": "vlm"
    }
  ]
}
```

## XFCE class taxonomy (starter)

- `button_save`
- `button_cancel`
- `input_filename`
- `input_text`
- `menu_item`
- `tab`
- `checkbox`
- `radio`
- `icon_file`
- `icon_folder`
- `window_titlebar`
- `dialog`

Adjust in `configs/classes.yaml` as needed.

## Conversion targets

- YOLO object detection: normalized `class x_center y_center width height`.
- Florence training pairs:
  - Input: screenshot image
  - Target text: compact element description list
