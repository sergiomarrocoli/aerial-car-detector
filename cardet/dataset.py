"""Turn hand-corrected tiles (from the labelling page) into an Ultralytics detection dataset.

Every car becomes a fixed square box on its centre. The hand-made boxes are tight and rotated, the detector's are loose
and upright, so their sizes disagree; the centre is the part everyone got right, and the later fill step finds the car's
real outline from the pixels anyway. Tiles are enlarged 2x because the detector works much better on bigger cars.
"""
import json
from pathlib import Path

from PIL import Image

BOX_PX = 20     # original pixels (a 4.5 m car is ~18 px long at 0.25 m/px)
SCALE = 2
VAL_TILES = ("t03", "t08")


def load_tiles(folder):
    """[(name, image path, [(cx, cy)])] for every tile marked done."""
    folder = Path(folder)
    out = []
    for f in sorted((folder / "labels").glob("*.json")):
        saved = json.loads(f.read_text())
        if saved["done"]:
            out.append((f.stem, folder / "tiles" / f"{f.stem}.jpg", [(b["cx"], b["cy"]) for b in saved["boxes"]]))
    return out


def yolo_lines(centres, size, scale=SCALE, box=BOX_PX):
    """YOLO label lines (class cx cy w h, normalised) for centres in original pixels."""
    side = size * scale
    lines = []
    for cx, cy in centres:
        x1, y1 = max(0, (cx - box / 2) * scale), max(0, (cy - box / 2) * scale)
        x2, y2 = min(side, (cx + box / 2) * scale), min(side, (cy + box / 2) * scale)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / side:.6f} {(y1 + y2) / 2 / side:.6f} {(x2 - x1) / side:.6f} {(y2 - y1) / side:.6f}")
    return lines


def build(folder, out, val=VAL_TILES):
    out = Path(out)
    counts = {"train": 0, "val": 0}
    for name, image, centres in load_tiles(folder):
        split = "val" if name in val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        im = Image.open(image).convert("RGB")
        im.resize((im.width * SCALE, im.height * SCALE), Image.LANCZOS).save(out / "images" / split / f"{name}.jpg", quality=95)
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(yolo_lines(centres, im.width)) + "\n")
        counts[split] += len(centres)
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: car\n")
    return counts


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1], sys.argv[2]))
