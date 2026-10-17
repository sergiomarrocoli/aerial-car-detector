import numpy as np

from cardet.detect import detect, nms, plausible, window_origins


def car(x, y, size=20, conf=0.5):
    return (x, y, x + size, y + size, conf)


def test_windows_cover_the_image_with_overlap_and_end_flush():
    starts = window_origins(2000, window=768, overlap=64)
    assert starts[0] == 0 and starts[-1] == 2000 - 768
    assert all(b - a <= 704 for a, b in zip(starts, starts[1:]))
    assert window_origins(500, window=768) == [0]


def test_nms_keeps_the_most_confident_of_overlapping_boxes():
    kept = nms([car(10, 10, conf=0.4), car(12, 11, conf=0.9), car(100, 100, conf=0.3)])
    assert [b[4] for b in kept] == [0.9, 0.3]


def test_implausible_boxes_are_dropped():
    assert plausible(car(0, 0, 20))
    assert not plausible(car(0, 0, 3))                      # speck
    assert not plausible(car(0, 0, 80))                     # roof
    assert not plausible((0, 0, 40, 10, 0.5))               # long strip (aspect 4)


def test_detect_shifts_window_boxes_to_image_coordinates_and_merges_the_overlap():
    rgb = np.zeros((1000, 1000, 3), np.uint8)
    starts = window_origins(1000)                           # [0, 232] on each axis
    origins = iter([(x, y) for y in starts for x in starts])

    def predict(window):                                    # a car at image (730..750, 740..760), seen by every window holding it
        x, y = next(origins)
        if x <= 730 and 750 <= x + 768 and y <= 740 and 760 <= y + 768:
            return [(730 - x, 740 - y, 750 - x, 760 - y, 0.7)]
        return []

    (box,) = detect(rgb, predict)
    assert abs(box["cx"] - 740) < 0.01 and abs(box["cy"] - 750) < 0.01
    assert box["a"] == 0 and box["conf"] == 0.7


def test_boxes_centred_on_buildings_are_dropped():
    rgb = np.zeros((300, 300, 3), np.uint8)
    buildings = np.zeros((300, 300), np.uint8)
    buildings[100:140, 100:140] = 1
    boxes = detect(rgb, lambda w: [car(110, 110), car(200, 200)], buildings)
    assert [(round(b["cx"]), round(b["cy"])) for b in boxes] == [(210, 210)]


def test_boxes_from_numpy_models_serialise_to_json():
    import json
    rgb = np.zeros((300, 300, 3), np.uint8)
    boxes = detect(rgb, lambda w: [tuple(np.float32(v) for v in car(50, 50))])
    json.dumps(boxes)
