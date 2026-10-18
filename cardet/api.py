"""The one call other programs need: take an aerial image, get it back with the cars painted out.

    from cardet.api import remove_cars
    cleaned, boxes = remove_cars(rgb, buildings=mask)       # rgb: HxWx3 uint8 at about 25 cm/px

`buildings` is an optional uint8 mask (HxW, 1 on building footprints): detections centred on buildings are dropped, because
roof clutter looks like cars. `conf` is the confidence floor: 0.3 trades a few missed cars for few false paint-overs.
"""
from . import NANO_WEIGHTS
from .detect import detect, yolo_predictor
from .fill import fill_cars

CONF = 0.3


def remove_cars(rgb, buildings=None, conf=CONF, weights=None, predict=None):
    """(cleaned image, [box dicts]). `predict` replaces the model (for tests); otherwise `weights` or the packaged nano model."""
    predict = predict or yolo_predictor(weights or NANO_WEIGHTS, conf=conf)
    boxes = [b for b in detect(rgb, predict, buildings) if b["conf"] >= conf]
    return fill_cars(rgb, boxes), boxes
