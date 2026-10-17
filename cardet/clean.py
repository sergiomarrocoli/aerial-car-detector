"""Remove cars from a bundle's imagery.

python3 -m cardet.clean path/to/bundle --weights data/models/yolocar-11n.pt --out clean.jpg [--conf 0.15]
Detects cars (upscaled 2x, buildings masked), paints them out, and writes the cleaned image.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .blobs import building_mask
from .detect import detect, yolo_predictor
from .fill import fill_cars

Image.MAX_IMAGE_PIXELS = None


def clean_bundle(bundle, weights, conf=0.15, tight=True):
    bundle = Path(bundle)
    meta = json.loads((bundle / "meta.json").read_text())
    rgb = np.asarray(Image.open(bundle / "imagery.jpg").convert("RGB"))
    h, w = rgb.shape[:2]
    features = json.loads((bundle / "buildings.geojson").read_text())["features"]
    buildings = building_mask(features, (h, w), w / meta["size_m"][0])
    boxes = [b for b in detect(rgb, yolo_predictor(weights, conf=conf), buildings) if b["conf"] >= conf]
    return rgb, boxes, fill_cars(rgb, boxes, tight=tight)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--box", action="store_true", help="mask the whole box instead of only the car's own pixels")
    a = ap.parse_args()
    _, boxes, cleaned = clean_bundle(a.bundle, a.weights, a.conf, not a.box)
    Image.fromarray(cleaned).save(a.out, quality=95)
    print(f"{len(boxes)} cars painted out -> {a.out}")
