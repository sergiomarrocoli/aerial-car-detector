import numpy as np

from cardet import NANO_WEIGHTS
from cardet.api import remove_cars


def test_the_nano_weights_ship_with_the_package():
    assert NANO_WEIGHTS.exists() and 5_000_000 < NANO_WEIGHTS.stat().st_size < 7_000_000


def test_confident_detections_are_painted_out_and_weak_ones_are_not():
    rgb = np.full((400, 400, 3), 90, np.uint8)
    rgb[100:107, 100:118] = (200, 30, 30)       # a red car on grey
    rgb[300:307, 300:318] = (30, 30, 200)       # a blue one the model is unsure about
    cleaned, boxes = remove_cars(rgb, predict=lambda window: [(99, 93.5, 119, 113.5, 0.9), (299, 293.5, 319, 313.5, 0.1)])   # x1, y1, x2, y2, conf
    assert [round(b["conf"], 1) for b in boxes] == [0.9]
    assert abs(int(cleaned[103, 109, 0]) - 90) < 25     # the red car is gone
    assert tuple(cleaned[303, 309]) == (30, 30, 200)     # the doubtful one is left alone
