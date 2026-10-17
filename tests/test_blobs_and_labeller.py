import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from cardet.blobs import building_mask, find_candidates  # noqa: E402
from cardet.labeller import make_handler  # noqa: E402
from cardet.prepare import pick_tiles  # noqa: E402


def road(h=120, w=160):
    return np.full((h, w, 3), 110, np.uint8)


def paint_car(img, cx, cy, length=18, width=7, colour=(200, 30, 30)):
    img[cy - width // 2: cy + width // 2 + 1, cx - length // 2: cx + length // 2 + 1] = colour


def test_a_car_on_the_road_is_found_with_its_size_and_direction():
    img = road()
    paint_car(img, 80, 60)
    (box,) = find_candidates(img)
    assert abs(box["cx"] - 80) < 2 and abs(box["cy"] - 60) < 2
    assert 16 <= box["l"] <= 20 and 6 <= box["w"] <= 9
    assert min(box["a"], 180 - box["a"]) < 5          # long axis is horizontal


def test_a_car_pointing_down_the_image_has_angle_near_90():
    img = road()
    img[50:69, 77:84] = (240, 240, 240)
    (box,) = find_candidates(img)
    assert abs(box["a"] - 90) < 5


def test_plain_road_and_thin_lane_markings_give_nothing():
    img = road()
    img[58:60, :] = 250                                  # a 2 px lane line
    assert find_candidates(img) == []


def test_a_blob_on_grass_is_not_a_car():
    img = np.full((120, 160, 3), (40, 140, 40), np.uint8)
    paint_car(img, 80, 60)
    assert find_candidates(img) == []


def test_blobs_under_the_building_mask_are_skipped():
    img = road()
    paint_car(img, 80, 60)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[40:80, 60:100] = 1
    assert find_candidates(img, mask) == []


def test_building_mask_rasterises_footprints_with_y_north():
    square = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]]]}}
    mask = building_mask([square], (100, 100), 1.0, margin=1)
    assert mask[85, 15] == 1 and mask[15, 15] == 0       # y=15 north is row 85 from the top


def test_pick_tiles_takes_the_densest_cells_best_first():
    boxes = [{"cx": 10, "cy": 10}] * 2 + [{"cx": 300, "cy": 10}] * 5 + [{"cx": 400, "cy": 400}]
    assert pick_tiles(boxes, 768, 768, 256, 2) == [(256, 0), (0, 0)]


@pytest.fixture
def labeller(tmp_path):
    (tmp_path / "tiles").mkdir()
    (tmp_path / "tiles" / "t01.jpg").write_bytes(b"jpeg")
    (tmp_path / "tiles.json").write_text(json.dumps({"px_per_m": 4, "tiles": [{"name": "t01", "candidates": [{"cx": 1}]}]}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_labels_are_saved_and_come_back_with_the_tiles(labeller):
    first = json.load(urllib.request.urlopen(f"{labeller}/tiles.json"))
    assert first["tiles"][0]["saved"] is None
    body = json.dumps({"boxes": [{"cx": 5}], "done": True}).encode()
    urllib.request.urlopen(urllib.request.Request(f"{labeller}/labels/t01", data=body, method="POST"))
    again = json.load(urllib.request.urlopen(f"{labeller}/tiles.json"))
    assert again["tiles"][0]["saved"] == {"boxes": [{"cx": 5}], "done": True}
    assert urllib.request.urlopen(f"{labeller}/tiles/t01.jpg").read() == b"jpeg"


def test_label_names_cannot_escape_the_folder(labeller):
    req = urllib.request.Request(f"{labeller}/labels/..%2Fx", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req)
    assert err.value.code == 400


def test_pick_tiles_prefers_cells_that_are_mostly_tarmac():
    boxes = [{"cx": 10, "cy": 10}] * 6 + [{"cx": 300, "cy": 10}] * 4
    surface = np.zeros((256, 512), bool)
    surface[:, 256:] = True                              # right cell is tarmac, left is garden
    assert pick_tiles(boxes, 512, 256, 256, 1, surface) == [(256, 0)]
