from __future__ import annotations

import csv
import json
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import sumolib

ROOT = Path(__file__).resolve().parents[2]
NETWORK_PATH = ROOT / "sumo" / "network" / "bengaluru.net.xml"
RAW_STATIONS_CSV = ROOT / "data" / "raw" / "bengaluru_charging_stations.csv"
SQLITE_DB = ROOT / "data" / "db.sqlite3"
SUMO_STATIONS_XML = ROOT / "sumo" / "stations" / "chargers.add.xml"

MIN_STATIONS = 500
MIN_SYNTHETIC_SPACING_M = 300.0

REAL_SOURCE = "real"
SYNTHETIC_SOURCE = "synthetic"

random.seed(42)


@dataclass
class StationRecord:
    station_id: str
    name: str
    lat: float
    lon: float
    edge_id: str
    lane_id: str
    lane_position: float
    source: str
    num_ports: int
    price_per_kwh: float
    max_grid_load_kw: float


def validate_inputs() -> None:
    if not NETWORK_PATH.exists():
        raise FileNotFoundError(
            "Missing SUMO network file sumo/network/bengaluru.net.xml. "
            "Please re-run Step A to generate or place the real network file."
        )
    if not RAW_STATIONS_CSV.exists():
        raise FileNotFoundError(
            "Missing real station CSV data at data/raw/bengaluru_charging_stations.csv. "
            "Please re-run Step B or provide the real CSV file."
        )
    if RAW_STATIONS_CSV.stat().st_size == 0:
        raise ValueError(
            "The station CSV file data/raw/bengaluru_charging_stations.csv is empty. "
            "Please re-run Step B to obtain real charging station data."
        )


def load_real_stations() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with RAW_STATIONS_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                lat = float(row.get("lat", 0.0) or row.get("latitude", 0.0))
                lon = float(row.get("lon", 0.0) or row.get("longitude", 0.0))
            except ValueError:
                continue
            if lat == 0.0 or lon == 0.0:
                continue
            records.append(
                {
                    "station_id": str(row.get("station_id") or row.get("id") or f"real_{len(records)+1}"),
                    "name": str(row.get("name", "Bengaluru Charging Station")),
                    "lat": lat,
                    "lon": lon,
                    "num_ports": max(1, int(row.get("num_ports", row.get("capacity", 1)) or 1)),
                    "price_per_kwh": float(row.get("price_per_kwh", row.get("price", 0)) or 0) or 0.0,
                }
            )
    if not records:
        raise ValueError(
            "No valid records found in data/raw/bengaluru_charging_stations.csv. "
            "Please ensure the CSV includes station_id, name, lat, lon, and port/pricing fields."
        )
    return records


def build_db() -> sqlite3.Connection:
    SQLITE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS charging_stations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            edge_id TEXT NOT NULL,
            lane_id TEXT NOT NULL,
            lane_position REAL NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            source TEXT NOT NULL CHECK(source IN ('real','synthetic')),
            num_ports INTEGER NOT NULL,
            price_per_kwh REAL NOT NULL,
            max_grid_load_kw REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_charging_stations_edge ON charging_stations(edge_id);
        """
    )
    conn.commit()
    return conn


def snap_station_to_network(net: sumolib.net.Net, lat: float, lon: float) -> tuple[str, str, float]:
    x, y = net.convertLonLat2XY(lon, lat)
    neighboring_lanes = net.getNeighboringLanes(x, y, 100.0)
    if not neighboring_lanes:
        raise RuntimeError(f"Could not snap station at ({lat},{lon}) to the SUMO network")
    lane_id = neighboring_lanes[0][0]
    lane = net.getLane(lane_id)
    lane_position = max(1.0, min(lane.getLength() - 1.0, lane.getLength() / 2.0))
    return lane.getEdge().getID(), lane_id, lane_position


def distance_squared(a: StationRecord, b: StationRecord) -> float:
    return (a.lat - b.lat) ** 2 + (a.lon - b.lon) ** 2


def sample_synthetic_stations(net: sumolib.net.Net, existing: list[StationRecord]) -> list[StationRecord]:
    all_edges = list(net.getEdges())
    generated: list[StationRecord] = []
    attempts = 0
    while len(existing) + len(generated) < MIN_STATIONS and attempts < 5000:
        edge = random.choice(all_edges)
        if edge.getLaneNumber() == 0:
            attempts += 1
            continue
        lane = edge.getLane(0)
        lane_position = random.uniform(20.0, max(20.0, lane.getLength() - 20.0))
        point = lane.getShape()[int(len(lane.getShape()) / 2)] if lane.getShape() else (0.0, 0.0)
        lat, lon = net.convertXY2LonLat(*point)
        candidate = StationRecord(
            station_id=f"synthetic_{len(existing) + len(generated) + 1}",
            name=f"Synthetic Station {len(existing) + len(generated) + 1}",
            lat=lat,
            lon=lon,
            edge_id=edge.getID(),
            lane_id=lane.getID(),
            lane_position=lane_position,
            source=SYNTHETIC_SOURCE,
            num_ports=random.choices([2, 3, 4, 5, 6, 7, 8], weights=[30, 25, 20, 10, 7, 5, 3])[0],
            price_per_kwh=round(random.uniform(6.0, 15.0), 2),
            max_grid_load_kw=round(random.uniform(50.0, 150.0), 1),
        )
        too_close = False
        for station in existing + generated:
            if ((station.lat - candidate.lat) ** 2 + (station.lon - candidate.lon) ** 2) ** 0.5 * 111000.0 < MIN_SYNTHETIC_SPACING_M:
                too_close = True
                break
        if too_close:
            attempts += 1
            continue
        generated.append(candidate)
        attempts += 1
    if len(existing) + len(generated) < MIN_STATIONS:
        raise RuntimeError("Could not generate enough synthetic stations with the requested spacing")
    return generated


def persist_stations(conn: sqlite3.Connection, stations: list[StationRecord]) -> None:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM charging_stations")
    for station in stations:
        cursor.execute(
            "INSERT INTO charging_stations (id, name, edge_id, lane_id, lane_position, lat, lon, source, num_ports, price_per_kwh, max_grid_load_kw) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                station.station_id,
                station.name,
                station.edge_id,
                station.lane_id,
                station.lane_position,
                station.lat,
                station.lon,
                station.source,
                station.num_ports,
                station.price_per_kwh,
                station.max_grid_load_kw,
            ),
        )
    conn.commit()


def write_sumo_additional(stations: list[StationRecord]) -> None:
    SUMO_STATIONS_XML.parent.mkdir(parents=True, exist_ok=True)
    lines = ["<additional>"]
    for station in stations:
        lines.append(
            f'  <chargingStation id="{station.station_id}" lane="{station.lane_id}" startPos="{station.lane_position:.1f}" power="{station.max_grid_load_kw:.1f}" chargeInTransit="false"/>'
        )
    lines.append("</additional>")
    SUMO_STATIONS_XML.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(stations: list[StationRecord]) -> None:
    total = len(stations)
    source_counts = Counter(station.source for station in stations)
    ports = Counter(station.num_ports for station in stations)

    print(f"Total stations: {total}")
    print(f"Real stations: {source_counts['real']}")
    print(f"Synthetic stations: {source_counts['synthetic']}")
    print("Port count histogram:")
    for ports_count, count in sorted(ports.items()):
        print(f"  {ports_count} ports: {count}")


def main() -> dict[str, Any]:
    validate_inputs()
    real_records = load_real_stations()
    net = sumolib.net.readNet(str(NETWORK_PATH))

    station_records: list[StationRecord] = []
    for record in real_records:
        edge_id, lane_id, lane_position = snap_station_to_network(net, record["lat"], record["lon"])
        station_records.append(
            StationRecord(
                station_id=str(record["station_id"]),
                name=str(record["name"]),
                lat=float(record["lat"]),
                lon=float(record["lon"]),
                edge_id=edge_id,
                lane_id=lane_id,
                lane_position=lane_position,
                source=REAL_SOURCE,
                num_ports=max(1, int(record["num_ports"])),
                price_per_kwh=round(float(record["price_per_kwh"]) if record["price_per_kwh"] else random.uniform(6.0, 15.0), 2),
                max_grid_load_kw=round(random.uniform(40.0, 120.0), 1),
            )
        )

    synthetic_stations = sample_synthetic_stations(net, station_records)
    station_records.extend(synthetic_stations)

    conn = build_db()
    persist_stations(conn, station_records)
    conn.close()

    write_sumo_additional(station_records)
    print_summary(station_records)
    return {
        "database": SQLITE_DB,
        "stations_xml": SUMO_STATIONS_XML,
        "total_stations": len(station_records),
        "real_stations": sum(1 for station in station_records if station.source == REAL_SOURCE),
        "synthetic_stations": sum(1 for station in station_records if station.source == SYNTHETIC_SOURCE),
    }


if __name__ == "__main__":
    main()
