from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "osm" / "bangalore_subarea.osm"
OUT.parent.mkdir(parents=True, exist_ok=True)

bbox = "77.590,12.930,77.650,12.990"
url = f"https://overpass-api.de/api/map?bbox={bbox}"
req = Request(url, headers={"User-Agent": "SmartEVNetworkFetcher/1.0"})

print(f"Fetching OSM network for bbox {bbox}")
try:
    with urlopen(req, timeout=60) as response:
        content = response.read()
    OUT.write_bytes(content)
    print(f"Saved OSM extract to {OUT}")
except Exception as exc:
    print(f"Overpass fetch failed: {exc}")
    print("Please manually export from https://www.openstreetmap.org/export using bbox:")
    print(bbox)
    print("and save it as data/osm/bangalore_subarea.osm")
