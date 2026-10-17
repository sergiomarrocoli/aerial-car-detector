"""Turn hand-corrected tiles (from the labelling page) into an Ultralytics detection dataset.

Every car becomes a fixed square box on its centre. The hand-made boxes are tight and rotated, the detector's are loose
and upright, so their sizes disagree; the centre is the part everyone got right, and the later fill step finds the car's
real outline from the pixels anyway. Tiles are enlarged 2x because the detector works much better on bigger cars.
"""
import json
import math
from pathlib import Path

from PIL import Image

BOX_PX = 20     # original pixels (a 4.5 m car is ~18 px long at 0.25 m/px)
LARGE_PX = 28   # a labelled vehicle longer than this (a bus, a lorry) keeps its real size instead of the fixed box
SCALE = 2
VAL_TILES = ("t03", "t08", "b03", "b06", "e03", "w01")   # two Newtown, two Bath, one Shrewsbury, one Porthmadog


def load_tiles(folder):
    """[(name, image path, [(cx, cy)])] for every tile marked done."""
    folder = Path(folder)
    out = []
    for f in sorted((folder / "labels").glob("*.json")):
        saved = json.loads(f.read_text())
        if saved["done"]:
            out.append((f.stem, folder / "tiles" / f"{f.stem}.jpg", [(b["cx"], b["cy"]) for b in saved["boxes"]]))
    return out


def box_size(b, box=BOX_PX, large=LARGE_PX):
    """(width, height) in original px for a saved label: the fixed box, or the real upright extent of a large vehicle."""
    a = math.radians(b.get("a", 0))
    hx = abs(b["l"] / 2 * math.cos(a)) + abs(b["w"] / 2 * math.sin(a))
    hy = abs(b["l"] / 2 * math.sin(a)) + abs(b["w"] / 2 * math.cos(a))
    if max(b["l"], b["w"]) > large:
        return max(box, 2 * hx), max(box, 2 * hy)
    return box, box


def load_labels(folder):
    """{tile name: (image path, [saved boxes])} for every tile marked done."""
    folder = Path(folder)
    out = {}
    for f in sorted((folder / "labels").glob("*.json")):
        saved = json.loads(f.read_text())
        if saved["done"]:
            out[f.stem] = (folder / "tiles" / f"{f.stem}.jpg", saved["boxes"])
    return out


def yolo_lines(centres, size, scale=SCALE, box=BOX_PX, sizes=None):
    """YOLO label lines (class cx cy w h, normalised) for centres in original pixels; `sizes` gives each box's (w, h)."""
    side = size * scale
    lines = []
    for i, (cx, cy) in enumerate(centres):
        bw, bh = sizes[i] if sizes else (box, box)
        x1, y1 = max(0, (cx - bw / 2) * scale), max(0, (cy - bh / 2) * scale)
        x2, y2 = min(side, (cx + bw / 2) * scale), min(side, (cy + bh / 2) * scale)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / side:.6f} {(y1 + y2) / 2 / side:.6f} {(x2 - x1) / side:.6f} {(y2 - y1) / side:.6f}")
    return lines


def rotate_centres(centres, size):
    """Centres after rotating a size x size image 90 degrees anticlockwise (PIL's ROTATE_90): (x, y) -> (y, size - x)."""
    return [(y, size - x) for x, y in centres]


def build(folder, out, val=VAL_TILES, rot90=True):
    """Write the dataset. With `rot90`, each training tile also appears turned 90 degrees: with the online left-right and
    up-down flips that gives all 8 orientations. Fixed square boxes make this exact, unlike Ultralytics' own rotation,
    which would grow every box."""
    out = Path(out)
    counts = {"train": 0, "val": 0}
    for name, (image, boxes) in load_labels(folder).items():
        centres = [(b["cx"], b["cy"]) for b in boxes]
        sizes = [box_size(b) for b in boxes]
        split = "val" if name in val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        im = Image.open(image).convert("RGB")
        im.resize((im.width * SCALE, im.height * SCALE), Image.LANCZOS).save(out / "images" / split / f"{name}.jpg", quality=95)
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(yolo_lines(centres, im.width, sizes=sizes)) + "\n")
        counts[split] += len(centres)
        if rot90 and split == "train":
            turned = im.transpose(Image.ROTATE_90)
            turned.resize((im.width * SCALE, im.height * SCALE), Image.LANCZOS).save(out / "images" / split / f"{name}_r90.jpg", quality=95)
            (out / "labels" / split / f"{name}_r90.txt").write_text("\n".join(yolo_lines(rotate_centres(centres, im.width), im.width, sizes=[(h, w) for w, h in sizes])) + "\n")
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n")
    return counts


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1], sys.argv[2]))
