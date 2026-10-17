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

## Model tuning

### Data

40 tiles of 768 px (0.25 m per pixel), 3,721 cars, corrected by hand in the labelling page. `yolocar`'s boxes were the
starting point; 60% of the final cars (2,224) had to be added by hand, so its boxes covered about 40% of the cars once
false ones were deleted.

| Area | Tiles | Cars |
|---|---|---|
| Newtown (`t`) | 16 | 1,230 |
| Bath (`b`) | 12 | 1,197 |
| Shrewsbury (`e`) | 6 | 984 |
| Porthmadog (`w`) | 6 | 310 |

Held out for testing: `t03 t08 b03 b06 e03 w01` (6 tiles, 527 cars, all four areas). The other 34 tiles (3,194 cars)
train the model. Labels and tile origins are in `labels/newtown/`; the tile images are cut again from a civil-war
bundle by `cardet.prepare`.

Every car is turned into a fixed 20 px square on its centre (`cardet/dataset.py`): the hand-made boxes are tight and
angled, the detector's loose and upright, so only the centre is trusted. Tiles are enlarged 2x (1536 px) because
detectors do much better on bigger cars. Each training tile also appears turned 90 degrees, and Ultralytics flips
left-right and up-down, so every tile is seen in all 8 orientations. Free-angle rotation is off: it would grow the boxes.

### Training

| | |
|---|---|
| Start from | `yolocar` (YOLO11n architecture, 5.5 MB), one class |
| Epochs / image size / batch | 40 / 1536 px / 2 |
| Optimiser | AdamW, learning rate 0.0005 |
| Augmentation | flips, 90 degree copies, Ultralytics defaults (mosaic, colour, scale, shift) |
| Hardware | Apple GPU (MPS), about 2 h 10 min wall clock with 4 crash/resume cycles |
| Result | `runs/detect/runs/ycA/weights/best.pt` (5.7 MB, not committed) |

Command: `python3 -m cardet.dataset data/newtown data/yolo` then
`python3 -m cardet.train data/yolo data/models/yolocar-11n.pt name=ycA epochs=40 lr0=0.0005`.
On the Apple GPU Ultralytics randomly crashes with a shape-mismatch error in its label assigner (epochs 20, 21, 31, 38
here, never the same place). `python3 -m cardet.train resume <last.pt>` carries on from the last full epoch, so nothing
is lost. It is not a data problem, and a CUDA GPU should not hit it (cause not confirmed).

![loss curve and held-out accuracy](docs/training-curves.png)

Left, the loss curve (log scale): training and held-out loss fall together from about 5.5 to 3.8, with no sign of
overfitting. Right, held-out precision, recall and mAP50 per epoch. The dips (epochs 9, 16, 19, 28 to 30) are mostly
validation noise at a high learning rate and just after restarts; the curve is flat from epoch 31 and ends at mAP50
0.82 (best epoch 31; mAP50-95 0.415).

### Score on the 527 held-out cars

A detection is right if its centre is within 2 m (8 px) of a labelled car; buildings are masked; each car is matched once
(`python3 -m cardet.score data/newtown <weights>`).

| Confidence | `yolocar` as downloaded: precision / recall / F1 | Fine-tuned (`ycA`): precision / recall / F1 |
|---|---|---|
| 0.1 | 0.83 / 0.61 / 0.70 | 0.70 / 0.84 / 0.76 |
| 0.2 | 0.89 / 0.50 / 0.64 | 0.80 / 0.81 / **0.81** |
| 0.3 | 0.91 / 0.40 / 0.56 | 0.87 / 0.76 / **0.81** |
| 0.4 | 0.93 / 0.33 / 0.49 | 0.91 / 0.67 / 0.77 |
| 0.5 | 0.96 / 0.25 / 0.39 | 0.94 / 0.54 / 0.69 |

Best F1 goes from 0.70 to 0.81. At about the same precision (0.80 vs 0.83) recall rises from 0.61 to 0.81.

### Earlier attempts (first 8 Newtown tiles only; 543 train cars, 136 held out, all Newtown)

| Weights | Best F1 |
|---|---|
| `yolocar` as downloaded | 0.80 (flattered: those labels began from its boxes) |
| `yolocar` fine-tuned, lr 0.002, 60 epochs | 0.75 |
| `yolocar` fine-tuned, backbone frozen, lr 0.0003, 30 epochs | 0.77 |
| standard `yolo11n` (COCO) fine-tuned | 0.68 |
| standard `yolo11s` (COCO) fine-tuned | 0.75 |

543 cars was not enough to beat the downloaded model. With 3,194 it is clearly better. A COCO `yolo11s` run on the full
data crashed at epoch 4 on the Apple GPU (about 4 min per epoch) and was dropped; it is worth running on a cloud GPU.

### Caveats

- The test tiles come from the same four areas as the training tiles (different tiles, same streets and imagery
  style), so the score says little about places never seen. Try a new town before trusting it elsewhere.
- Labels are only as complete as the labelling was; a few real cars are unlabelled, which lowers measured precision.
- Cars cut by the edge of a tile are labelled as clipped boxes; `detect.py` drops half-cars at inner window edges.

## Plan

1. Run the detector at 2x (`cardet/detect.py`), buildings masked, confidence about 0.2 to 0.3 with the fine-tuned weights
2. Fill the cars out of the imagery (`cardet/fill.py`, `python3 -m cardet.clean`); before and after for the downloaded
   model: `docs/before-after-centre.jpg`, `docs/before-after-station.jpg`
3. Re-clean Newtown with the fine-tuned weights and compare (done, confidence 0.25): 1,552 cars painted out against 1,075
   with the downloaded model, and the leftover dark and shadowed cars in the car parks and the station road are gone.
   `docs/compare-centre.jpg`, `docs/compare-station.jpg`: original, downloaded model, fine-tuned (left to right). Soft
   dark patches remain where shadows were filled, and some rail wagons are still partly removed
4. Decide how civil-war uses it (pipeline stage vs a one-off step) and publish the weights as a release

Weights are not committed: `data/`, `runs/` and `*.pt` are ignored. Fetch `yolocar` with
`curl -L -o data/models/yolocar-11n.pt https://huggingface.co/wuhp/yolocar/resolve/main/car-75e-11n.pt`.

## More places

`cardet/fetch_area.py` fetches a 1 km box of Esri imagery and OpenStreetMap buildings for named places (Bristol, Cardiff,
Manchester, Edinburgh, Swindon, Leeds, Brighton, Aberystwyth so far; add a line to try another) and writes a small bundle
that `cardet.prepare` cuts into tiles, pre-labelled by the current best weights. Where the model fails: per-tile
scores on the held-out set showed dense city streets with buses and shadows (Bath `b03` F1 0.59, Shrewsbury `e03` 0.72)
are much worse than Newtown (0.83 to 0.87), so new tiles lean towards those. Vehicles over 28 px (buses, lorries) now keep
their real size in training instead of the fixed 20 px box, and the detector keeps boxes up to 75 px.

## Hand-labelling tiles

```
python3 -m cardet.prepare path/to/bundle --out data/newtown --weights data/models/yolocar-11n.pt [--prefix b --count 12 --skip 0]
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
