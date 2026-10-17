"""Cut a bundle's imagery into tiles for hand-labelling cars, with a detector's boxes pre-drawn.

The tiles with the most candidate cars are picked (no two overlapping), so the labelling time goes on cars.

Usage: python3 -m cardet.prepare data/bundles/area_newtown --out data/newtown --weights data/models/yolocar-11n.pt
Then:  python3 -m cardet.labeller data/newtown

With --weights the boxes come from that detector (at confidence 0.1, so false boxes are easier to delete than missed
cars are to add); without it they come from the model-free blob finder.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from .detect import detect, yolo_predictor
from .blobs import building_mask, find_candidates, grey_surface

Image.MAX_IMAGE_PIXELS = None


CONFIDENT_CONTRAST = 55  # gardens and roof edges give many weak blobs; real cars stand out


def pick_tiles(boxes, width, height, size, count, surface=None):
    """Grid cells of `size` px with the most confident boxes, best first. Cells never overlap.
    `surface` (optional bool mask of grey tarmac) scales each cell's score by its share of tarmac, so gardens lose."""
    cols, rows = width // size, height // size
    tally = {}
    for b in boxes:
        if b.get("contrast", CONFIDENT_CONTRAST) < CONFIDENT_CONTRAST:
            continue
        c, r = int(b["cx"] // size), int(b["cy"] // size)
        if c < cols and r < rows:
            tally[(c, r)] = tally.get((c, r), 0) + 1
    if surface is not None:
        for (c, r) in tally:
            tally[(c, r)] *= surface[r * size:(r + 1) * size, c * size:(c + 1) * size].mean()
    best = sorted(tally, key=lambda k: -tally[k])[:count]
    return [(c * size, r * size) for c, r in best]


CONFIDENT_CONF = 0.3  # for ranking tiles by detector boxes


def prepare(bundle, out, size=768, count=8, weights=None):
    import numpy as np
    bundle, out = Path(bundle), Path(out)
    meta = json.loads((bundle / "meta.json").read_text())
    image = Image.open(bundle / "imagery.jpg").convert("RGB")
    rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    px_per_m = w / meta["size_m"][0]
    features = json.loads((bundle / "buildings.geojson").read_text())["features"]
    buildings = building_mask(features, (h, w), px_per_m)
    if weights:
        boxes = detect(rgb, yolo_predictor(weights), buildings)
        for b in boxes:                              # rank tiles on the confident ones, like the blob contrast
            b["contrast"] = CONFIDENT_CONTRAST if b["conf"] >= CONFIDENT_CONF else 0
    else:
        boxes = find_candidates(rgb, buildings)
    surface = grey_surface(rgb) & (buildings == 0)

    (out / "tiles").mkdir(parents=True, exist_ok=True)
    tiles = []
    for n, (x0, y0) in enumerate(pick_tiles(boxes, w, h, size, count, surface), 1):
        name = f"t{n:02d}"
        image.crop((x0, y0, x0 + size, y0 + size)).save(out / "tiles" / f"{name}.jpg", quality=95)
        inside = [{**b, "cx": b["cx"] - x0, "cy": b["cy"] - y0, "src": "auto"} for b in boxes
                  if x0 <= b["cx"] < x0 + size and y0 <= b["cy"] < y0 + size]
        tiles.append({"name": name, "x0": x0, "y0": y0, "size": size, "candidates": inside})
    (out / "tiles.json").write_text(json.dumps({"px_per_m": px_per_m, "tiles": tiles}))
    return tiles


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--weights", help="Ultralytics weights for the detector; omit to use the blob finder")
    a = ap.parse_args()
    for t in prepare(a.bundle, a.out, a.size, a.count, a.weights):
        print(f"{t['name']}: {len(t['candidates'])} boxes at ({t['x0']}, {t['y0']})")
