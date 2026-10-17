# aerial-car-detector

A small detector for **cars in 25 cm aerial photos**, trained to paint parked cars out of the imagery used by
[civil-war](https://github.com/sergiomarrocoli/civil-war) (they otherwise end up flat on the road).

## Why this exists

Pretrained aerial YOLOs (DOTA) find 0 to 12 of the ~150 cars in a Newtown test crop: cars are only ~18 x 7 px and the
imagery differs from what they were trained on. So we fine-tune a small model (`yolov8n-obb`, 6 MB, runs on CPU) on
public labelled car data, resized to our pixel size, and check it on hand-labelled Newtown tiles.

## Plan

1. `data/raw/`: COWC detection patches (Potsdam, Selwyn), downloaded from
   https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/detection/ (not committed)
2. Convert the labels to YOLO format, resized to ~0.25 m per pixel
3. Fine-tune `yolov8n-obb`
4. Score it on Newtown tiles you corrected with the labelling page, against the colour-blob baseline
5. Publish the weights as a GitHub release; civil-war's pipeline downloads them on first use

## Hand-labelling Newtown tiles (the test set)

```
python3 -m cardet.prepare path/to/civil-war/data/bundles/area_newtown --out data/newtown
python3 -m cardet.labeller data/newtown        # http://127.0.0.1:8790
```

Click a red box to delete it, click empty ground to add a car, drag along a car to add it pointing the right way.
Press D when every car in a tile has a box. `cardet/blobs.py` is the model-free baseline that pre-draws the boxes.

## Setup

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```
