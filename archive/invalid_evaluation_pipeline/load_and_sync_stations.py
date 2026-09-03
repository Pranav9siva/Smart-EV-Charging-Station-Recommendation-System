"""Load real stations from CSV, synthesize to 500, write to stations DB, and regenerate stations.add.xml

Usage: python scripts/load_and_sync_stations.py
"""
from __future__ import annotations

import csv
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.simulation.bangalore_scenario import _snap_station_to_lane
DB_PATH = ROOT / "data" / "stations" / "stations.sqlite"
CSV_PATH = ROOT / "data" / "raw" / "bengaluru_charging_stations.csv"
SIM_DIR = ROOT / "simulations" / "bangalore"
NETWORK_PATH = SIM_DIR / "network.net.xml"


def ensure_schema(conn: sqlite3.Connection) -> None:
    # tables already exist in the provided sqlite; create if missing
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS stations (
            station_id TEXT PRIMARY KEY,
            name TEXT,
            lat REAL,
            lon REAL,
            operator TEXT,
            address TEXT,
            grid_zone_id TEXT,
            data_source TEXT CHECK(data_source IN ('real','simulated'))
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ports (
            port_id TEXT PRIMARY KEY,
            station_id TEXT REFERENCES stations(station_id),
            connector_type TEXT,
            power_kw REAL,
            status TEXT,
            status_updated_at TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS occupancy_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            port_id TEXT REFERENCES ports(port_id),
            status TEXT,
            changed_at TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT REFERENCES stations(station_id),
            price_per_kwh REAL,
            tariff_period TEXT,
            recorded_at TIMESTAMP
        )
        """
    )
    conn.commit()


def load_csv_records(path: Path) -> List[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                lat = float(row.get("lat", 0.0) or 0.0)
                lon = float(row.get("lon", 0.0) or 0.0)
            except ValueError:
                continue
            if lat == 0.0 or lon == 0.0:
                continue
            records.append({
                "station_id": row.get("station_id") or f"real_{len(records)+1}",
                "name": row.get("name", "Charging Station"),
                "lat": lat,
                "lon": lon,
                "num_ports": max(1, int(row.get("num_ports") or 1)),
                "power_kw": float(row.get("power_kw") or 22.0),
                "operator": row.get("operator") or "",
                "address": row.get("address") or "",
                "source": row.get("source") or "real",
            })
    return records


def synthesize(records: List[dict], target: int = 500) -> List[dict]:
    out = list(records)
    if len(out) >= target:
        return out[:target]
    random.seed(0)
    idx = 0
    while len(out) < target:
        base = random.choice(records)
        jlat = base["lat"] + random.uniform(-0.02, 0.02)
        jlon = base["lon"] + random.uniform(-0.02, 0.02)
        idx += 1
        out.append(
            {
                "station_id": f"sim_{idx}",
                "name": f"Sim Station {idx}",
                "lat": jlat,
                "lon": jlon,
                "num_ports": random.choice([1, 2, 4, 6]),
                "power_kw": random.choice([22.0, 50.0, 75.0]),
                "operator": "simulator",
                "address": "",
                "source": "simulated",
            }
        )
    return out


def write_db(conn: sqlite3.Connection, records: List[dict]) -> None:
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    for rec in records:
        cur.execute(
            "INSERT OR REPLACE INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?,?,?,?,?,?,?,?)",
            (rec["station_id"], rec["name"], rec["lat"], rec["lon"], rec.get("operator", ""), rec.get("address", ""), None, rec.get("source", "simulated")),
        )
        # create ports
        for p in range(int(rec.get("num_ports", 1))):
            port_id = f"{rec['station_id']}_p{p+1}"
            cur.execute(
                "INSERT OR REPLACE INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?,?,?,?,?,?)",
                (port_id, rec["station_id"], "CCS", rec.get("power_kw", 22.0), "Available", now),
            )
    conn.commit()


def regenerate_additional_file(conn: sqlite3.Connection, net_path: Path, out_path: Path, lookup_path: Path) -> None:
    import sumolib

    net = sumolib.net.readNet(str(net_path))
    cur = conn.cursor()
    rows = cur.execute("SELECT station_id, name, lat, lon, operator FROM stations").fetchall()
    stations = []
    for row in rows:
        station_id, name, lat, lon, operator = row
        try:
            lane_id, start_pos = _snap_station_to_lane(net, lat, lon)
        except RuntimeError:
            continue
        stations.append({
            "station_id": station_id,
            "name": name,
            "num_ports": len(conn.execute("SELECT port_id FROM ports WHERE station_id = ?", (station_id,)).fetchall()),
            "power_kw": conn.execute("SELECT power_kw FROM ports WHERE station_id = ? LIMIT 1", (station_id,)).fetchone()[0],
            "lane": lane_id,
            "start_pos": round(start_pos, 1),
            "distance_km": 1.0,
            "travel_time_min": 2.0,
        })

    # write stations.add.xml
    lines = ["<additional>"]
    for s in stations:
        lines.append(f'  <chargingStation id="{s["station_id"]}" lane="{s["lane"]}" startPos="{s["start_pos"]}" power="{s["power_kw"]}" chargeInTransit="false"/>')
    lines.append("</additional>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # write lookup
    import json

    lookup = {s["station_id"]: s for s in stations}
    lookup_path.write_text(json.dumps(lookup, indent=2), encoding="utf-8")


def main() -> None:
    conn = sqlite3.connect(str(DB_PATH))
    ensure_schema(conn)
    records = load_csv_records(CSV_PATH)
    if not records:
        print("No CSV records found; aborting")
        return
    all_records = synthesize(records, target=500)
    write_db(conn, all_records)
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    regenerate_additional_file(conn, NETWORK_PATH, SIM_DIR / "stations.add.xml", ROOT / "data" / "stations" / "stations_lookup.json")
    print("Loaded and synced stations; total:", len(all_records))


if __name__ == "__main__":
    main()
