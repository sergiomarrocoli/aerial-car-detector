"""Find car-sized blobs in aerial imagery, with no model: a car is a small compact patch that differs from the
grey surface around it. Used to pre-draw boxes for hand-correction and as a baseline for the trained detector.

A box is {cx, cy, l, w, a, contrast}: centre in pixels, length and width in pixels, `a` the long axis in degrees
(image coordinates, 0 = pointing right, 90 = pointing down), in [0, 180), and how strongly the blob stands out.
"""
import cv2
import numpy as np

BACKGROUND_WINDOW = 31       # px; larger than a car, so the median is the road under it
MIN_CONTRAST = 30            # max channel difference from the background
MAX_SATURATION = 70          # background must be grey (road, car park), not grass
MIN_AREA, MAX_AREA = 30, 300  # px² at ~0.25 m/px (a car is about 130)
LENGTH_PX, WIDTH_PX = (7, 28), (4, 12)
MIN_FILL = 0.55              # blob area / its bounding rectangle
BUILDING_MARGIN_PX = 7


def building_mask(features, shape, px_per_m, margin=BUILDING_MARGIN_PX):
    """Rasterise building footprints (GeoJSON features in local metres, y north) into a uint8 mask."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for f in features:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            pts = np.array([[x * px_per_m, h - y * px_per_m] for x, y in poly[0]], np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return cv2.dilate(mask, np.ones((margin, margin), np.uint8)) if margin > 1 else mask


def _box(points):
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(points)
    if rw < rh:
        rw, rh, angle = rh, rw, angle + 90
    return {"cx": float(cx), "cy": float(cy), "l": float(rw), "w": float(rh), "a": float(angle % 180)}


def _contrast_of(contrast, stats_row, mask):
    x, y, w, h = stats_row[:4]
    return float(contrast[y:y + h, x:x + w][mask].mean())


def grey_surface(rgb):
    """Boolean mask of pixels whose surroundings are grey (tarmac, car parks), not grass or trees."""
    return cv2.cvtColor(cv2.medianBlur(rgb, BACKGROUND_WINDOW), cv2.COLOR_RGB2HSV)[..., 1] < MAX_SATURATION


def find_candidates(rgb, buildings=None):
    """Return the boxes of car-like blobs in an HxWx3 uint8 image; `buildings` is an optional mask to skip."""
    bg = cv2.medianBlur(rgb, BACKGROUND_WINDOW)
    contrast = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    on_grey = cv2.cvtColor(bg, cv2.COLOR_RGB2HSV)[..., 1] < MAX_SATURATION
    m = (contrast > MIN_CONTRAST) & on_grey
    if buildings is not None:
        m &= buildings == 0
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    boxes = []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        inside = labels[stats[i, 1]:stats[i, 1] + stats[i, 3], stats[i, 0]:stats[i, 0] + stats[i, 2]] == i
        ys, xs = np.nonzero(inside)
        pts = np.stack([xs + stats[i, 0], ys + stats[i, 1]], 1).astype(np.float32)
        box = _box(pts)
        if LENGTH_PX[0] <= box["l"] <= LENGTH_PX[1] and WIDTH_PX[0] <= box["w"] <= WIDTH_PX[1] \
                and area / (box["l"] * box["w"]) > MIN_FILL:
            box["contrast"] = _contrast_of(contrast, stats[i], inside)
            boxes.append(box)
    return boxes
