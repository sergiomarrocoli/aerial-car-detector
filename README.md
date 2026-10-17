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

## Fine-tuning result (8 Newtown tiles labelled by hand, 679 cars; t03 and t08 held out, 136 cars)

Scored by matching each detection to a labelled car within 2 m, buildings masked (`python3 -m cardet.score`):

| Weights | Best F1 | Precision / recall at that point |
|---|---|---|
| `yolocar` as downloaded, conf 0.1 | **0.80** | 0.83 / 0.78 |
| fine-tuned, lr 0.002, 60 epochs | 0.75 | 0.76 / 0.75 (conf 0.2) |
| fine-tuned gently (backbone frozen, lr 0.0003, 30 epochs) | 0.77 | 0.77 / 0.77 (conf 0.4) |

Six training tiles were not enough to beat the downloaded model, and the held-out set is small (about +-0.04). The labels
also began from `yolocar`'s own boxes, which flatters it. Most false detections are skylights and roof features, which
the building footprints remove. Labels and tile origins are saved in `labels/newtown/`.

## Plan

1. Run `yolocar` at 2x (`cardet/detect.py`), buildings masked, confidence around 0.1 to 0.2
2. Fill the cars out of the imagery (`cardet/fill.py`, `python3 -m cardet.clean`): about 1,100 cars on Newtown in 20 s.
   Before and after: `docs/before-after-centre.jpg`, `docs/before-after-station.jpg`. Cars it misses (dark ones,
   some in shadow) stay; the fill is slightly soft and keeps most painted lines
3. More labelled tiles from other areas only if the fill looks wrong where the detector misses
4. Publish the settings (and weights, if we ever beat the download) for civil-war's pipeline to use

Weights are not committed: `data/` and `runs/` are ignored. Fetch `yolocar` with
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
