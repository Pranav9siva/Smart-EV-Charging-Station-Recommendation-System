from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.engine import RecommendationEngine, StationCandidate
DB = ROOT / "recommendations.sqlite"
LOOKUP = ROOT / "data" / "stations" / "stations_lookup.json"
FLEET_MANIFEST = ROOT / "data" / "fleet" / "fleet_manifest.json"


def init_db() -> None:
    with sqlite3.connect(DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_log (
                run_id TEXT NOT NULL,
                vehicle_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                soc_pct_at_decision REAL NOT NULL,
                station_id TEXT NOT NULL,
                score REAL NOT NULL,
                travel_time_min REAL NOT NULL,
                queue_wait_min REAL NOT NULL,
                charge_time_min REAL NOT NULL,
                chosen INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def main() -> None:
    init_db()
    stations_lookup = json.loads(LOOKUP.read_text(encoding="utf-8")) if LOOKUP.exists() else {}
    fleet_manifest = json.loads(FLEET_MANIFEST.read_text(encoding="utf-8")) if FLEET_MANIFEST.exists() else []
    engine = RecommendationEngine()

    for vehicle in fleet_manifest[:3]:
        soc_pct = float(vehicle.get("starting_soc_pct", 50.0))
        candidates = []
        for station_id, station_data in stations_lookup.items():
            candidates.append(
                StationCandidate(
                    station_id=station_id,
                    distance_km=2.0 + len(candidates) * 0.5,
                    travel_time_min=4.0,
                    free_ports=max(1, int(station_data.get("num_ports", 1))),
                    total_ports=max(1, int(station_data.get("num_ports", 1))),
                    power_kw=float(station_data.get("power_kw", 22.0)),
                )
            )
        recommendation = engine.score_station(
            {"soc_pct": soc_pct, "battery_kwh": float(vehicle.get("battery_kwh", 40.0)), "consumption_wh_per_km": float(vehicle.get("consumption_wh_per_km", 180.0))},
            candidates[0],
        )
        with sqlite3.connect(DB) as conn:
            conn.execute(
                "INSERT INTO recommendation_log (run_id, vehicle_id, timestamp, soc_pct_at_decision, station_id, score, travel_time_min, queue_wait_min, charge_time_min, chosen) VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?)",
                ("phase2_demo", vehicle["vehicle_id"], soc_pct, recommendation.station_id, recommendation.score, recommendation.travel_time_min, recommendation.queue_wait_min, recommendation.charge_time_min, 1),
            )
            conn.commit()
        print(f"{vehicle['vehicle_id']}: SoC {soc_pct}% -> {recommendation.station_id}")


if __name__ == "__main__":
    main()
