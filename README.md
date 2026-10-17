# aerial-car-detector

A small detector for **cars in 25 cm aerial photos**, trained to paint parked cars out of the imagery used by
[civil-war](https://github.com/sergiomarrocoli/civil-war) (they otherwise end up flat on the road).

## What we found

Cars are ~18 x 7 px in our 25 cm imagery. On a Newtown crop with ~150 cars:

| Detector | Found |
|---|---|
| Pretrained DOTA `yolov8n-obb` / `yolo11n-obb` | 0 to 12 |
| Colour blobs (`cardet/blobs.py`) | most red/white cars, many false boxes |
| VHR-10 `yolov8n` ([bluelabel](https://huggingface.co/bluelabel/satellite-equipment-detection-yolov8n-vhr10)) | 30 at best, upscaled 3x |
| OWLv2 (public Space, "a photo of a car", 2x) | most cars, many duplicates and false boxes; not pursued |
| **`yolo11n` [wuhp/yolocar](https://huggingface.co/wuhp/yolocar)** (5.5 MB) | 100 at conf 0.3, 121 at 0.2, upscaled 2x; roughly half to two thirds of cars, few false boxes |

COWC (the usual aerial car dataset) was checked and set aside: its downloads are classification patches or one point
per car, not boxes.

## Plan

1. Run `yolocar` at 2x over the imagery (`cardet/detect.py`), at confidence 0.1: easier to delete false boxes than to add misses
2. Correct a few Newtown tiles in the labelling page (below)
3. Fine-tune `yolocar` on the corrected tiles, holding one tile out; compare against the 0.2-confidence baseline
4. Publish the weights as a GitHub release; civil-war's pipeline downloads them and paints the cars out

Weights are not committed: `data/` is ignored. Fetch `yolocar` with
`curl -L -o data/models/yolocar-11n.pt https://huggingface.co/wuhp/yolocar/resolve/main/car-75e-11n.pt`.

## Hand-labelling Newtown tiles (the test set)

```
python3 -m cardet.prepare path/to/civil-war/data/bundles/area_newtown --out data/newtown --weights data/models/yolocar-11n.pt
python3 -m cardet.labeller data/newtown        # http://127.0.0.1:8790
```

Boxes run from yellow (weak) to red (confident). Click a box to delete it, click empty ground to add a car, drag along
a car to add it pointing the right way. "Drop N weak" removes every dashed box under the confidence you set. Press D
when every car in a tile has a box. Omit `--weights` to pre-draw with the colour-blob finder instead.

## Setup

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```
