"""Score a detector on hand-labelled tiles: match each detection to a labelled car by centre distance."""
import numpy as np
from PIL import Image

from .dataset import SCALE
from .detect import detect, yolo_predictor

MATCH_PX = 8   # a detection within this many original pixels (2 m) of a labelled centre is that car


def match(dets, truth, radius=MATCH_PX):
    """Greedy by confidence. dets [(cx, cy, conf)], truth [(cx, cy)] -> list of (conf, is_true_positive), misses unmatched."""
    free = list(range(len(truth)))
    results = []
    for cx, cy, conf in sorted(dets, key=lambda d: -d[2]):
        best, best_d = None, radius
        for i in free:
            d = np.hypot(cx - truth[i][0], cy - truth[i][1])
            if d <= best_d:
                best, best_d = i, d
        if best is not None:
            free.remove(best)
        results.append((conf, best is not None))
    return results


def curve(results, n_truth, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Precision, recall and F1 at each confidence threshold."""
    rows = []
    for t in thresholds:
        kept = [tp for c, tp in results if c >= t]
        tp = sum(kept)
        p = tp / len(kept) if kept else 0.0
        r = tp / n_truth if n_truth else 0.0
        rows.append((t, p, r, 2 * p * r / (p + r) if p + r else 0.0))
    return rows


def evaluate(weights, tiles, conf=0.05, scale=SCALE, augment=False, iou=0.5):
    """tiles: [(image path, [(cx, cy)], building mask or None)] in original pixels. Returns the curve over all tiles together."""
    predict = yolo_predictor(weights, scale=scale, conf=conf, augment=augment, iou=iou)
    results, n_truth = [], 0
    for image, centres, buildings in tiles:
        boxes = detect(np.asarray(Image.open(image).convert("RGB")), predict, buildings)
        results += match([(b["cx"], b["cy"], b["conf"]) for b in boxes], centres)
        n_truth += len(centres)
    return curve(results, n_truth), n_truth
