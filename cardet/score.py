"""Print precision / recall / F1 of weights on the held-out tiles.

python3 -m cardet.score data/newtown weights.pt
Boxes centred on building footprints are dropped (using each tile's bundle), as in the real pipeline.
"""
import json
import sys
from pathlib import Path

from PIL import Image

from .blobs import building_mask
from .dataset import VAL_TILES, load_tiles
from .evaluate import evaluate

Image.MAX_IMAGE_PIXELS = None


def tile_masks(folder):
    """Building mask cropped to each tile, from the bundle recorded in tiles.json (tiles without one get no mask)."""
    masks, cache = {}, {}
    for t in json.loads((Path(folder) / "tiles.json").read_text())["tiles"]:
        bundle = t.get("bundle")
        if not bundle:
            continue
        if bundle not in cache:
            meta = json.loads((Path(bundle) / "meta.json").read_text())
            w, h = Image.open(Path(bundle) / "imagery.jpg").size
            features = json.loads((Path(bundle) / "buildings.geojson").read_text())["features"]
            cache[bundle] = building_mask(features, (h, w), w / meta["size_m"][0])
        x, y, size = t["x0"], t["y0"], t["size"]
        masks[t["name"]] = cache[bundle][y:y + size, x:x + size]
    return masks


if __name__ == "__main__":
    folder, weights = sys.argv[1], sys.argv[2]
    masks = tile_masks(folder)
    tiles = [(image, centres, masks.get(name)) for name, image, centres in load_tiles(folder) if name in VAL_TILES]
    rows, n = evaluate(weights, tiles)
    print(f"{weights}: {n} cars in {len(tiles)} held-out tiles" + (" (buildings masked)" if masks else ""))
    for t, p, r, f1 in rows:
        print(f"  conf>={t:.1f}  precision {p:.2f}  recall {r:.2f}  F1 {f1:.2f}")
