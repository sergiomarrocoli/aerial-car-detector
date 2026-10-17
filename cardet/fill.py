"""Paint detected cars out of aerial imagery.

All boxes are masked together before filling, so in a row of parked cars the fill comes from the tarmac around the row,
never from the neighbouring cars. Filled pixels get grain matching their surroundings, since a perfectly smooth patch
looks like a smudge.
"""
import cv2
import numpy as np

MARGIN_PX = 3        # beyond the box, for the car's edge and nearest shadow
MIN_HALF_PX = 10     # a car (any direction) fits within a 20 px square at 0.25 m/px
INPAINT_RADIUS = 6
GRAIN_SIGMA = 1.5    # blur used to split detail from colour when measuring grain


def car_mask(shape, boxes, margin=MARGIN_PX):
    """uint8 mask (1 = paint out): a square around each box centre, as big as the box plus `margin`."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for b in boxes:
        half = max(MIN_HALF_PX, max(b["l"], b["w"]) / 2) + margin
        x1, y1 = max(0, int(round(b["cx"] - half))), max(0, int(round(b["cy"] - half)))
        x2, y2 = min(w, int(round(b["cx"] + half))), min(h, int(round(b["cy"] + half)))
        mask[y1:y2, x1:x2] = 1
    return mask


def tight_mask(rgb, boxes, margin=2, contrast=22, core=6):
    """Mask only the car's own pixels: inside each box, those that differ from the tarmac around it, plus a small core.

    The tarmac colour is the median of a ring just outside the box, ignoring pixels inside other boxes, so in a row of
    cars it comes from the gaps between rows. Kerbs and painted lines the box merely touches are left alone.
    """
    h, w = rgb.shape[:2]
    boxed = car_mask(rgb.shape, boxes, margin=0)
    mask = np.zeros((h, w), np.uint8)
    for b in boxes:
        half = max(MIN_HALF_PX, max(b["l"], b["w"]) / 2)
        x1, y1, x2, y2 = (int(round(v)) for v in (b["cx"] - half, b["cy"] - half, b["cx"] + half, b["cy"] + half))
        X1, Y1, X2, Y2 = max(0, x1 - 4), max(0, y1 - 4), min(w, x2 + 4), min(h, y2 + 4)
        win = rgb[Y1:Y2, X1:X2].astype(np.int16)
        ring = np.ones(win.shape[:2], bool)
        ring[max(0, y1 - Y1):y2 - Y1, max(0, x1 - X1):x2 - X1] = False
        ring &= boxed[Y1:Y2, X1:X2] == 0
        inside = np.zeros(win.shape[:2], bool)
        inside[max(0, y1 - Y1):y2 - Y1, max(0, x1 - X1):x2 - X1] = True
        if ring.sum() < 20:                                   # boxed in on all sides: fall back to the whole box
            mask[max(0, y1):y2, max(0, x1):x2] = 1
            continue
        background = np.median(win[ring], axis=0)
        car = (np.abs(win - background).max(axis=2) > contrast) & inside
        cy, cx = int(round(b["cy"])) - Y1, int(round(b["cx"])) - X1
        cv2.circle(car.view(np.uint8), (cx, cy), core, 1, -1)
        mask[Y1:Y2, X1:X2] |= car.astype(np.uint8)
    k = 2 * margin + 1
    return cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), np.ones((k, k), np.uint8))


def _local_grain(rgb, mask, window=41):
    """Per-pixel standard deviation of fine detail, measured only on pixels outside the mask."""
    detail = rgb.astype(np.float32) - cv2.GaussianBlur(rgb, (0, 0), GRAIN_SIGMA).astype(np.float32)
    keep = (mask == 0).astype(np.float32)
    weight = cv2.blur(keep, (window, window)) + 1e-6
    power = cv2.blur((detail ** 2).mean(axis=2) * keep, (window, window)) / weight
    return np.sqrt(power)


def fill_cars(rgb, boxes, margin=MARGIN_PX, seed=0, tight=True):
    """Return a copy of an HxWx3 uint8 image with the boxed cars painted out (tight: only the car's own pixels)."""
    if not boxes:
        return rgb.copy()
    mask = tight_mask(rgb, boxes) if tight else car_mask(rgb.shape, boxes, margin)
    out = cv2.inpaint(rgb, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    rng = np.random.default_rng(seed)
    grain = _local_grain(rgb, mask)[..., None] * rng.standard_normal(rgb.shape).astype(np.float32)
    painted = out.astype(np.float32) + grain * mask[..., None]
    return np.clip(painted, 0, 255).astype(np.uint8)
