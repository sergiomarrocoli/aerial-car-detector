"""Fetch aerial imagery and building outlines for a place, as a small bundle that `cardet.prepare` can cut tiles from.

    python3 -m cardet.fetch_area data/areas norwich plymouth      (names from AREAS below; none given = all)
    python3 -m cardet.fetch_area data/areas --at 52.63 1.30 mytown

Each place is a 1 km box centred on a latitude and longitude. Imagery is Esri World Imagery (zoom 19) resampled to
exactly 0.25 m/px (4000 x 4000 px); buildings come from OpenStreetMap via Overpass. Both are for development only: the
Esri imagery is not licensed for redistribution. The bundle holds `imagery.jpg`, `buildings.geojson` (building outlines
in metres east and north of the box's south-west corner) and `meta.json`.
"""
import argparse
import io
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

AREAS = {   # name: (latitude, longitude) of the centre
    "bristol": (51.4545, -2.5879),       # city centre: buses, one-way streets, shadows
    "cardiff": (51.4816, -3.1791),
    "manchester": (53.4840, -2.2370),    # dense city blocks
    "edinburgh": (55.9570, -3.2020),     # New Town terraces, on-street parking
    "swindon": (51.5700, -1.7900),       # estates and retail parks
    "leeds": (53.8200, -1.5800),         # terraced suburbs
    "brighton": (50.8300, -0.1400),
    "aberystwyth": (52.4140, -4.0830),   # small seaside town
    "norwich": (52.6309, 1.2974),        # unseen towns for the clean held-out tiles
    "plymouth": (50.3755, -4.1427),
}

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
ZOOM, TILE, M_PER_PX = 19, 256, 0.25
AGENT = {"User-Agent": "aerial-car-detector (development use)"}


def metres_per_degree(lat):
    """(metres per degree of latitude, metres per degree of longitude) at `lat`, from the WGS84 series."""
    p = math.radians(lat)
    return (111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p),
            111412.84 * math.cos(p) - 93.5 * math.cos(3 * p))


def to_local(lat, lon, centre, size_m):
    """Metres east and north of the south-west corner of the box centred on `centre`."""
    m_lat, m_lon = metres_per_degree(centre[0])
    return (lon - centre[1]) * m_lon + size_m / 2, (lat - centre[0]) * m_lat + size_m / 2


def mercator_px(lat, lon, zoom=ZOOM):
    """Pixel coordinates (x east, y south) in the world Web Mercator image at `zoom`."""
    scale = TILE * 2 ** zoom
    x = (np.asarray(lon, dtype="float64") + 180) / 360 * scale
    lat_r = np.radians(np.asarray(lat, dtype="float64"))
    y = (0.5 - np.log(np.tan(np.pi / 4 + lat_r / 2)) / (2 * np.pi)) * scale
    return x, y


def _tile(zxy):
    z, x, y = zxy
    for attempt in range(3):
        try:
            r = requests.get(ESRI.format(z=z, x=x, y=y), timeout=30, headers=AGENT)
            r.raise_for_status()
            return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise


def fetch_imagery(centre, size_m=1000):
    """RGB array, 0.25 m/px, north up, covering the box exactly."""
    n = round(size_m / M_PER_PX)
    cols, rows = np.meshgrid((np.arange(n) + 0.5) * M_PER_PX, size_m - (np.arange(n) + 0.5) * M_PER_PX)   # local metres
    m_lat, m_lon = metres_per_degree(centre[0])
    lat = centre[0] + (rows - size_m / 2) / m_lat
    lon = centre[1] + (cols - size_m / 2) / m_lon
    px, py = mercator_px(lat, lon)
    x0, x1, y0, y1 = int(px.min() // TILE), int(px.max() // TILE), int(py.min() // TILE), int(py.max() // TILE)
    coords = [(ZOOM, x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        tiles = list(pool.map(_tile, coords))
    mosaic = np.zeros(((y1 - y0 + 1) * TILE, (x1 - x0 + 1) * TILE, 3), np.uint8)
    for (_, x, y), img in zip(coords, tiles):
        mosaic[(y - y0) * TILE:(y - y0 + 1) * TILE, (x - x0) * TILE:(x - x0 + 1) * TILE] = img
    map_x = (px - x0 * TILE - 0.5).astype("float32")
    map_y = (py - y0 * TILE - 0.5).astype("float32")
    return cv2.remap(mosaic, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _overpass(query):
    errors = []
    for attempt in range(5):
        for url in OVERPASS:
            try:
                r = requests.post(url, data={"data": query}, timeout=90, headers=AGENT)
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{url}: {type(exc).__name__}")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("All Overpass endpoints failed: " + "; ".join(errors))


def rings(element):
    """Closed outlines [(lat, lon), ...] of an Overpass element: a building way, or the closed outer ways of a relation."""
    if element["type"] == "way":
        geoms = [element.get("geometry")]
    else:
        geoms = [m.get("geometry") for m in element.get("members", []) if m.get("role") == "outer" and m["type"] == "way"]
    out = []
    for g in geoms:
        pts = [(p["lat"], p["lon"]) for p in g or []]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            out.append(pts)
    return out


def fetch_buildings(centre, size_m=1000):
    """GeoJSON FeatureCollection of building outlines in local metres (origin = south-west corner, y north)."""
    m_lat, m_lon = metres_per_degree(centre[0])
    s, n = centre[0] - size_m / 2 / m_lat, centre[0] + size_m / 2 / m_lat
    w, e = centre[1] - size_m / 2 / m_lon, centre[1] + size_m / 2 / m_lon
    query = (f'[out:json][timeout:60];(way["building"]({s},{w},{n},{e});relation["building"]({s},{w},{n},{e}););out geom;')
    feats = []
    for el in _overpass(query)["elements"]:
        for ring in rings(el):
            coords = [[round(v, 2) for v in to_local(lat, lon, centre, size_m)] for lat, lon in ring]
            feats.append({"type": "Feature", "properties": {"osm_id": f'{el["type"]}/{el["id"]}', "building": el.get("tags", {}).get("building")},
                          "geometry": {"type": "Polygon", "coordinates": [coords]}})
    return {"type": "FeatureCollection", "features": feats}


def fetch_area(name, lat, lon, out_root, size_m=1000):
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    centre = (lat, lon)
    Image.fromarray(fetch_imagery(centre, size_m)).save(out / "imagery.jpg", quality=92)
    buildings = fetch_buildings(centre, size_m)
    (out / "buildings.geojson").write_text(json.dumps(buildings, separators=(",", ":")))
    (out / "meta.json").write_text(json.dumps({"id": name, "centre": [lat, lon], "origin": [0, 0], "size_m": [size_m, size_m]}))
    return out, len(buildings["features"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("names", nargs="*", help="places from AREAS; none = all (or the one given with --at)")
    ap.add_argument("--at", nargs=2, type=float, metavar=("LAT", "LON"), help="fetch a new place centred here; the one name is its folder")
    a = ap.parse_args()
    if a.at:
        todo = {a.names[0]: tuple(a.at)}
    else:
        todo = {n: AREAS[n] for n in a.names or AREAS}
    for name, (lat, lon) in todo.items():
        out, count = fetch_area(name, lat, lon, a.root)
        print(f"{name}: {count} buildings -> {out}", flush=True)
