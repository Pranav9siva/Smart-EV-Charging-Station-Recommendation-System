from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "stations" / "stations.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

BENGALURU_BBOX = {
    "min_lat": 12.83,
    "min_lon": 77.45,
    "max_lat": 13.14,
    "max_lon": 77.75,
}

OPENCHARGEMAP_URL = (
    "https://api.openchargemap.io/v3/poi/?output=json&countrycode=IN&compact=true&verbose=false"
)

CONNECTOR_MAP = {
    "ccs2": "CCS2",
    "type2": "Type2",
    "chademo": "CHAdeMO",
    "gb-t": "GB-T",
    "ac": "Type2",
    "dc": "CCS2",
}

STATUS_SIMULATED = "simulated"
STATUS_REAL = "real"


@dataclass
class RawStation:
    station_id: str
    name: str
    lat: float
    lon: float
    operator: str
    address: str
    connector_types: list[str]
    power_kw: float
    num_ports: int
    data_source: str


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def similar_name(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def fetch_open_charge_map(api_key: str | None = None) -> list[RawStation]:
    query = {
        "boundingbox": f"({BENGALURU_BBOX['min_lat']},{BENGALURU_BBOX['min_lon']}),({BENGALURU_BBOX['max_lat']},{BENGALURU_BBOX['max_lon']})",
        "maxresults": "200",
    }
    url = OPENCHARGEMAP_URL + "&" + urlencode(query)
    headers = {"User-Agent": "SmartEVStationSeeder/1.0"}
    if api_key:
        headers["X-API-Key"] = api_key

    req = Request(url, headers=headers)
    with urlopen(req, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    stations: list[RawStation] = []
    for item in payload:
        address = item.get("AddressInfo", {})
        if not address:
            continue
        connector_types = []
        for connection in item.get("Connections", []):
            raw_type = str(connection.get("ConnectionType", {}).get("Title", "")).lower()
            if raw_type:
                connector_types.append(CONNECTOR_MAP.get(raw_type, raw_type.upper()))
        if not connector_types:
            connector_types = ["Type2"]

        stations.append(
            RawStation(
                station_id=str(item.get("ID", "")),
                name=item.get("AddressInfo", {}).get("Title", "Open Charge Map Station"),
                lat=float(address.get("Latitude", 0.0)),
                lon=float(address.get("Longitude", 0.0)),
                operator=item.get("OperatorInfo", {}).get("Title", "Unknown") or "Unknown",
                address=address.get("AddressLine1", "") or address.get("Town", "") or "",
                connector_types=connector_types,
                power_kw=float(item.get("UsageCost", 22.0) if isinstance(item.get("UsageCost"), (int, float)) else 22.0),
                num_ports=int(item.get("NumberOfPoints", 1) or 1),
                data_source=STATUS_REAL,
            )
        )
    return stations


def load_bee_yatra_csv(filepath: Path) -> list[RawStation]:
    stations: list[RawStation] = []
    if not filepath.exists():
        return stations

    with filepath.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                lat = float(row.get("lat", row.get("latitude", 0.0)) or 0.0)
                lon = float(row.get("lon", row.get("longitude", 0.0)) or 0.0)
            except ValueError:
                continue
            if not (BENGALURU_BBOX["min_lat"] <= lat <= BENGALURU_BBOX["max_lat"] and BENGALURU_BBOX["min_lon"] <= lon <= BENGALURU_BBOX["max_lon"]):
                continue
            connector_types = [row.get("connector_type", "Type2")]
            stations.append(
                RawStation(
                    station_id=str(row.get("station_id") or f"bee_{len(stations)}"),
                    name=str(row.get("name", "BEE EV Station")),
                    lat=lat,
                    lon=lon,
                    operator=str(row.get("operator", "BEE")) or "BEE",
                    address=str(row.get("address", "")),
                    connector_types=[CONNECTOR_MAP.get(c.strip().lower(), c.strip()) for c in connector_types if c],
                    power_kw=float(row.get("power_kw", 22.0) or 22.0),
                    num_ports=int(row.get("num_ports", 1) or 1),
                    data_source=STATUS_REAL,
                )
            )
    return stations


def deduplicate_stations(stations: Iterable[RawStation]) -> list[RawStation]:
    unique: list[RawStation] = []
    for station in stations:
        duplicate = False
        for existing in unique:
            if haversine(station.lat, station.lon, existing.lat, existing.lon) < 0.1 and similar_name(station.name, existing.name) > 0.75:
                duplicate = True
                break
        if not duplicate:
            unique.append(station)
    return unique


def create_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS stations (
            station_id TEXT PRIMARY KEY,
            name TEXT,
            lat REAL,
            lon REAL,
            operator TEXT,
            address TEXT,
            grid_zone_id TEXT,
            data_source TEXT CHECK(data_source IN ('real','simulated'))
        );
        CREATE TABLE IF NOT EXISTS ports (
            port_id TEXT PRIMARY KEY,
            station_id TEXT REFERENCES stations(station_id),
            connector_type TEXT,
            power_kw REAL,
            status TEXT CHECK(status IN ('Available','Preparing','Charging','SuspendedEV','SuspendedEVSE','Finishing','Reserved','Unavailable','Faulted')),
            status_updated_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT REFERENCES stations(station_id),
            price_per_kwh REAL,
            tariff_period TEXT CHECK(tariff_period IN ('solar_hour','non_solar_hour')),
            recorded_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS occupancy_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            port_id TEXT REFERENCES ports(port_id),
            status TEXT,
            changed_at TIMESTAMP
        );
        """
    )
    conn.commit()
    return conn


def create_ports_for_station(station: RawStation) -> list[tuple[str, str, float, str, str]]:
    ports = []
    connector_types = station.connector_types or ["Type2"]
    for idx in range(1, station.num_ports + 1):
        connector = connector_types[(idx - 1) % len(connector_types)]
        power_kw = station.power_kw if station.power_kw > 0 else (50.0 if connector in ("CCS2", "CHAdeMO", "GB-T") else 22.0)
        port_id = f"{station.station_id}_p{idx}"
        ports.append((port_id, station.station_id, connector, power_kw, "Available", datetime.now(timezone.utc).isoformat()))
    return ports


def seed_database(api_key: str | None = None, bee_csv: str | None = None, db_path: Path = DB_PATH) -> None:
    real_stations = []
    try:
        real_stations = fetch_open_charge_map(api_key=api_key)
    except URLError:
        print("Warning: Open Charge Map API unavailable; falling back to BEE Yatra data if provided.", file=sys.stderr)

    if bee_csv:
        bee_stations = load_bee_yatra_csv(Path(bee_csv))
        real_stations.extend(bee_stations)

    if not real_stations:
        print("No real station data found; creating a minimal simulated dataset.", file=sys.stderr)
        real_stations = [
            RawStation(
                station_id="sim_station_1",
                name="Bangalore Simulated Station 1",
                lat=12.95,
                lon=77.60,
                operator="Simulated Operator",
                address="Bangalore, KA",
                connector_types=["Type2"],
                power_kw=22.0,
                num_ports=2,
                data_source=STATUS_SIMULATED,
            )
        ]

    stations = deduplicate_stations(real_stations)
    conn = create_db(db_path)
    cursor = conn.cursor()

    for station in stations:
        grid_zone_id = f"zone_{int(station.lat*100)}_{int(station.lon*100)}"
        cursor.execute(
            "INSERT OR REPLACE INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (station.station_id, station.name, station.lat, station.lon, station.operator, station.address, grid_zone_id, station.data_source),
        )
        for port in create_ports_for_station(station):
            cursor.execute(
                "INSERT OR REPLACE INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                port,
            )

    conn.commit()
    print(f"Seeded {len(stations)} stations and {sum(len(create_ports_for_station(s)) for s in stations)} ports into {db_path}")
    conn.close()


def main() -> None:
    api_key = os.getenv("OPENCHARGEMAP_API_KEY")
    bee_csv = None
    if len(sys.argv) > 1:
        bee_csv = sys.argv[1]
    seed_database(api_key=api_key, bee_csv=bee_csv)


if __name__ == "__main__":
    main()
