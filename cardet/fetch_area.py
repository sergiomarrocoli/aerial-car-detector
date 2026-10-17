"""Fetch imagery and building outlines for new places, as small bundles that `cardet.prepare` can cut tiles from.

Uses civil-war's pipeline code (Esri World Imagery at ~0.25 m/px, OpenStreetMap buildings), so it must run with a Python
that has civil-war's pipeline requirements (rasterio, geopandas, pyproj). Run it as a script, not with -m:

    CIVIL_WAR=~/projects/civil-war python3 cardet/fetch_area.py data/areas bristol cardiff ...
Each place is a 1 km box centred on a point in the table below; add a line to try somewhere new.
"""
import json
import os
import sys
from pathlib import Path

AREAS = {   # name: (latitude, longitude) of the centre
    "bristol": (51.4545, -2.5879),       # city centre: buses, one-way streets, shadows
    "cardiff": (51.4816, -3.1791),
    "manchester": (53.4840, -2.2370),    # dense city blocks
    "edinburgh": (55.9570, -3.2020),     # New Town terraces, on-street parking
    "swindon": (51.5700, -1.7900),       # estates and retail parks
    "leeds": (53.8200, -1.5800),         # terraced suburbs
    "brighton": (50.8300, -0.1400),
    "aberystwyth": (52.4140, -4.0830),   # small seaside town
}


def fetch_area(name, lat, lon, out_root, civil_war, size_m=1000):
    sys.path.insert(0, str(Path(civil_war).expanduser()))
    import geopandas as gpd
    import rasterio
    from PIL import Image
    from pipeline.buildings import fetch_buildings
    from pipeline.bundle import _local_buildings
    from pipeline.imagery import fetch_imagery
    from pyproj import Transformer

    e, n = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True).transform(lon, lat)
    min_e, min_n = round((e - size_m / 2) / 25) * 25, round((n - size_m / 2) / 25) * 25       # snapped like the builder's boxes
    bbox = (min_e, min_n, min_e + size_m, min_n + size_m)
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)

    tif = fetch_imagery(bbox, out / "imagery.tif")
    with rasterio.open(tif) as ds:
        Image.fromarray(ds.read().transpose(1, 2, 0)).save(out / "imagery.jpg", quality=92)
    tif.unlink()

    raw = fetch_buildings(bbox, out / "buildings_raw.geojson")
    gdf = gpd.read_file(raw)
    (out / "buildings.geojson").write_text(json.dumps(_local_buildings(gdf, (min_e, min_n)), separators=(",", ":")))
    Path(raw).unlink()
    (out / "meta.json").write_text(json.dumps({"id": name, "bbox": list(bbox), "origin": [min_e, min_n], "size_m": [size_m, size_m]}))
    return out, len(gdf)


if __name__ == "__main__":
    root, names = sys.argv[1], sys.argv[2:] or list(AREAS)
    civil_war = os.environ.get("CIVIL_WAR", "~/projects/civil-war")
    for name in names:
        out, n = fetch_area(name, *AREAS[name], root, civil_war)
        print(f"{name}: {n} buildings -> {out}", flush=True)
