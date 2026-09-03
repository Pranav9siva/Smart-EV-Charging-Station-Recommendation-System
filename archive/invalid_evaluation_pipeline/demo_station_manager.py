from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.station_management.manager import StationManager


def main() -> None:
    db_path = ROOT / "data" / "stations" / "stations.sqlite"
    manager = StationManager(str(db_path))
    try:
        stations = manager.list_all_stations()
        print(f"stations_loaded={len(stations)}")
        if stations:
            sample = manager.get_station_state(stations[0]["station_id"])
            print("sample_station", sample["station_id"], sample.get("ports", [])[:2])
            manager.update_port_status(sample["ports"][0]["port_id"], "Charging")
            refreshed = manager.get_station_state(stations[0]["station_id"])
            print("updated_status", refreshed["ports"][0]["status"])
    finally:
        manager.close()


if __name__ == "__main__":
    main()
