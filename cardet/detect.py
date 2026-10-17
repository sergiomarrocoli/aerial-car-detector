"""Run a car detector over aerial imagery in overlapping windows, upscaled so small cars look bigger.

`detect` is model-agnostic: it takes `predict(window_rgb) -> [(x1, y1, x2, y2, conf)]` in window pixels, and handles
windowing, merging duplicates and dropping boxes that can't be cars. `yolo_predictor` wraps an Ultralytics model.

Boxes are upright {cx, cy, l, w, a=0, conf}: `l` is the x extent, `w` the y extent, in image pixels.
"""
import numpy as np

WINDOW_PX, OVERLAP_PX = 768, 64
SCALE = 2                    # cars are ~18 px at 0.25 m/px; detectors do much better at ~36 px
MIN_CONF = 0.1
SIZE_PX = (6, 40)            # a car box is 10-30 px; anything outside is a roof or a patch of ground
MAX_ASPECT = 3.0
NMS_IOU = 0.5
EDGE_PX = 3                  # a box this close to a window side is a car cut in half by the window, not a whole car


def yolo_predictor(weights, scale=SCALE, conf=MIN_CONF):
    from PIL import Image
    from ultralytics import YOLO
    model = YOLO(str(weights))

    def predict(window):
        h, w = window.shape[:2]
        size = (w * scale, h * scale)
        big = Image.fromarray(window).resize(size, Image.LANCZOS)
        r = model.predict(big, imgsz=max(size) // 32 * 32, conf=conf, iou=NMS_IOU, verbose=False)[0]
        xyxy, c = r.boxes.xyxy.cpu().numpy() / scale, r.boxes.conf.cpu().numpy()
        return [(*b, s) for b, s in zip(xyxy, c)]
    return predict


def window_origins(length, window=WINDOW_PX, overlap=OVERLAP_PX):
    """Start offsets along one axis so windows of `window` px cover `length` with `overlap`, the last flush to the edge."""
    if length <= window:
        return [0]
    step = window - overlap
    starts = list(range(0, length - window, step))
    return starts + [length - window]


def nms(boxes, iou=NMS_IOU):
    """Keep the highest-confidence box of any overlapping group. boxes: [(x1, y1, x2, y2, conf)]."""
    order = sorted(boxes, key=lambda b: -b[4])
    kept = []
    for b in order:
        if all(_iou(b, k) < iou for k in kept):
            kept.append(b)
    return kept


def _iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def plausible(b, size=SIZE_PX, max_aspect=MAX_ASPECT):
    w, h = b[2] - b[0], b[3] - b[1]
    return size[0] <= w <= size[1] and size[0] <= h <= size[1] and max(w, h) / min(w, h) <= max_aspect


def _cut_by_window(x1, y1, x2, y2, x0, y0, tw, th, w, h, edge=EDGE_PX):
    """True if the box touches a side of its window that is not the edge of the whole image: the overlap between windows
    means another window sees that car whole, so this partial detection is dropped."""
    return ((x1 < edge and x0 > 0) or (y1 < edge and y0 > 0)
            or (x2 > tw - edge and x0 + tw < w) or (y2 > th - edge and y0 + th < h))


def detect(rgb, predict, buildings=None, window=WINDOW_PX, overlap=OVERLAP_PX):
    """Detect cars in an HxWx3 uint8 image. `buildings` (optional mask) drops boxes centred on a building."""
    h, w = rgb.shape[:2]
    found = []
    for y0 in window_origins(h, window, overlap):
        for x0 in window_origins(w, window, overlap):
            tile = np.ascontiguousarray(rgb[y0:y0 + window, x0:x0 + window])
            th, tw = tile.shape[:2]
            for x1, y1, x2, y2, c in predict(tile):
                if _cut_by_window(x1, y1, x2, y2, x0, y0, tw, th, w, h):
                    continue
                found.append(tuple(float(v) for v in (x1 + x0, y1 + y0, x2 + x0, y2 + y0, c)))
    boxes = []
    for b in nms([b for b in found if plausible(b)]):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if buildings is not None and buildings[min(h - 1, int(cy)), min(w - 1, int(cx))]:
            continue
        boxes.append({"cx": cx, "cy": cy, "l": b[2] - b[0], "w": b[3] - b[1], "a": 0.0, "conf": b[4]})
    return boxes
