import numpy as np

from cardet.fetch_area import metres_per_degree, mercator_px, rings, to_local


def test_a_degree_of_latitude_is_about_111_km_and_longitude_shrinks_with_latitude():
    lat_m, lon_m = metres_per_degree(52.0)
    assert 111_200 < lat_m < 111_350
    assert 68_000 < lon_m < 69_000
    assert metres_per_degree(0)[1] > 111_000 > metres_per_degree(60)[1]


def test_local_coordinates_put_the_centre_in_the_middle_and_north_up():
    centre = (52.0, 1.0)
    assert to_local(*centre, centre, 1000) == (500, 500)
    x, y = to_local(52.001, 1.001, centre, 1000)
    assert x > 500 and y > 500
    assert abs((y - 500) - metres_per_degree(52.0)[0] * 0.001) < 1e-6


def test_mercator_pixels_grow_east_and_south():
    x1, y1 = mercator_px(52.0, 1.0)
    x2, y2 = mercator_px(51.999, 1.001)
    assert x2 > x1 and y2 > y1
    assert abs((x2 - x1) - 256 * 2 ** 19 / 360 * 0.001) < 1e-3


def test_only_closed_outlines_become_rings():
    closed = [{"lat": 0, "lon": 0}, {"lat": 0, "lon": 1}, {"lat": 1, "lon": 1}, {"lat": 0, "lon": 0}]
    assert rings({"type": "way", "geometry": closed}) == [[(0, 0), (0, 1), (1, 1), (0, 0)]]
    assert rings({"type": "way", "geometry": closed[:3]}) == []
    relation = {"type": "relation", "members": [{"type": "way", "role": "outer", "geometry": closed},
                                                {"type": "way", "role": "inner", "geometry": closed}]}
    assert len(rings(relation)) == 1
