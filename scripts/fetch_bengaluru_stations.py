from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RAW_DIR / "bengaluru_charging_stations.csv"

BBOX = {
    "min_lat": 12.83,
    "min_lon": 77.45,
    "max_lat": 13.14,
    "max_lon": 77.75,
}
OPENCHARGEMAP_URL = "https://api.openchargemap.io/v3/poi/"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_open_charge_map(api_key: str | None = None) -> list[dict[str, Any]]:
    params = {
        "output": "json",
        "countrycode": "IN",
        "boundingbox": f"({BBOX['min_lat']},{BBOX['min_lon']}),({BBOX['max_lat']},{BBOX['max_lon']})",
        "maxresults": "500",
        "compact": "true",
        "verbose": "false",
    }
    url = OPENCHARGEMAP_URL + "?" + urlencode(params)
    headers = {"User-Agent": "SmartEVStationFetcher/1.0"}
    if api_key:
        headers["X-API-Key"] = api_key

    req = Request(url, headers=headers)
    with urlopen(req, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    stations: list[dict[str, Any]] = []
    for item in payload:
        address = item.get("AddressInfo") or {}
        if not address:
            continue
        stations.append(
            {
                "station_id": f"ocm_{item.get('ID', '')}",
                "name": item.get("Title") or "OpenChargeMap Station",
                "lat": float(address.get("Latitude", 0.0)) if address.get("Latitude") is not None else 0.0,
                "lon": float(address.get("Longitude", 0.0)) if address.get("Longitude") is not None else 0.0,
                "num_ports": int(item.get("NumberOfPoints", 1) or 1),
                "power_kw": float(item.get("UsageCost", 22.0) or 22.0),
                "operator": item.get("OperatorInfo", {}).get("Title", "Unknown") or "Unknown",
                "address": address.get("AddressLine1", "") or address.get("Town", ""),
                "source": "real",
            }
        )
    return stations


def fetch_overpass_stations() -> list[dict[str, Any]]:
    query = f"[out:json];node[\"amenity\"=\"charging_station\"]({BBOX['min_lat']},{BBOX['min_lon']},{BBOX['max_lat']},{BBOX['max_lon']});out;"
    req = Request(OVERPASS_URL, data=query.encode("utf-8"), headers={"User-Agent": "SmartEVStationFetcher/1.0"})
    with urlopen(req, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    stations: list[dict[str, Any]] = []
    for node in payload.get("elements", []):
        tags = node.get("tags", {})
        lat = float(node.get("lat", 0.0) or 0.0)
        lon = float(node.get("lon", 0.0) or 0.0)
        stations.append(
            {
                "station_id": f"osm_{node.get('id', 0)}",
                "name": tags.get("name", "OSM Charging Station"),
                "lat": lat,
                "lon": lon,
                "num_ports": int(tags.get("capacity", 1) or 1),
                "power_kw": 22.0,
                "operator": tags.get("operator", "Unknown"),
                "address": tags.get("addr:street", ""),
                "source": "real",
            }
        )
    return stations


def normalize_station(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "station_id": str(record["station_id"]),
        "name": str(record.get("name", "Charging Station")),
        "lat": float(record.get("lat", 0.0) or 0.0),
        "lon": float(record.get("lon", 0.0) or 0.0),
        "num_ports": max(1, int(record.get("num_ports", 1) or 1)),
        "power_kw": max(1.0, float(record.get("power_kw", 22.0) or 22.0)),
        "operator": str(record.get("operator", "Unknown")) or "Unknown",
        "address": str(record.get("address", "")) or "",
        "source": str(record.get("source", "real")),
    }


def save_csv(records: list[dict[str, Any]]) -> None:
    if not records:
        raise RuntimeError("No station records to write to data/raw/bengaluru_charging_stations.csv")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["station_id", "name", "lat", "lon", "num_ports", "power_kw", "operator", "address", "source"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main() -> None:
    api_key = os.getenv("OPENCHARGEMAP_API_KEY")
    stations = []

    if api_key:
        try:
            stations = fetch_open_charge_map(api_key=api_key)
        except Exception as exc:  # pragma: no cover
            print(f"OpenChargeMap fetch failed: {exc}")

    if not stations:
        print("Falling back to Overpass API for charging station data.")
        stations = fetch_overpass_stations()

    normalized = [normalize_station(record) for record in stations if record.get("lat") and record.get("lon")]
    if not normalized:
        raise RuntimeError(
            "Could not fetch any charging stations from OpenChargeMap or Overpass. "
            "Set OPENCHARGEMAP_API_KEY or check your internet connection."
        )

    save_csv(normalized)
    print(f"Saved {len(normalized)} charging stations to {OUT_CSV}")


if __name__ == "__main__":
    main()
