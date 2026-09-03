from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "stations" / "bangalore_stations.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

bbox = "77.590,12.930,77.650,12.990"


def fetch_open_charge_map() -> list[dict]:
    api_key = os.getenv("CHARGEMAP_API_KEY")
    if not api_key:
        return []
    url = (
        "https://api.openchargemap.io/v3/poi/?output=json&countrycode=IN"
        f"&boundingbox=(12.990,77.590),(12.930,77.650)&maxresults=200&compact=true&verbose=false"
    )
    req = Request(url, headers={"X-API-Key": api_key})
    with urlopen(req, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data


def fallback_overpass() -> list[dict]:
    overpass_url = (
        "https://overpass-api.de/api/interpreter?data="
        + quote("[out:json];node[\"amenity\"=\"charging_station\"](12.930,77.590,12.990,77.650);out;")
    )
    with urlopen(overpass_url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    stations = []
    for node in payload.get("elements", []):
        tags = node.get("tags", {})
        stations.append(
            {
                "station_id": f"osm_{node.get('id', 0)}",
                "name": tags.get("name", "OpenStreetMap charging station"),
                "lat": node.get("lat", 0.0),
                "lon": node.get("lon", 0.0),
                "num_ports": int(tags.get("capacity", 1) or 1),
                "power_kw": 22.0,
                "connector_types": ["unknown"],
            }
        )
    return stations


items = fetch_open_charge_map()
if not items:
    items = fallback_overpass()

normalized = []
for item in items:
    if not item.get("AddressInfo"):
        continue
    address = item["AddressInfo"]
    normalized.append(
        {
            "station_id": str(item.get("ID", len(normalized))),
            "name": item.get("Title") or "Charging station",
            "lat": float(address.get("Latitude", 0.0)),
            "lon": float(address.get("Longitude", 0.0)),
            "num_ports": int(item.get("NumberOfPoints", 1) or 1),
            "power_kw": 22.0,
            "connector_types": ["AC"],
        }
    )

OUT.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
print(f"Wrote {len(normalized)} charging stations to {OUT}")
