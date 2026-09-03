from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import traci
from sumolib import checkBinary

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.seed_stations import create_db
from src.recommendation.engine import RecommendationEngine, StationCandidate
from src.station_management.occupancy_simulator import OccupancySimulator
from src.station_management.pricing import PricingManager

NETWORK = ROOT / "simulations" / "bangalore" / "network.net.xml"
STATIONS_XML = ROOT / "simulations" / "bangalore" / "stations.add.xml"
ROUTES_XML = ROOT / "simulations" / "bangalore" / "evs.rou.xml"
SUMO_CFG = ROOT / "simulations" / "bangalore" / "sim.sumocfg"
LOOKUP = ROOT / "data" / "stations" / "stations_lookup.json"
FLEET_MANIFEST = ROOT / "data" / "fleet" / "fleet_manifest.json"
DB = ROOT / "recommendations.sqlite"
STATION_DB = ROOT / "data" / "stations" / "stations.sqlite"

if not NETWORK.exists():
    raise FileNotFoundError(f"SUMO network missing: {NETWORK}")
if not STATIONS_XML.exists():
    raise FileNotFoundError(f"Stations file missing: {STATIONS_XML}")
if not ROUTES_XML.exists():
    raise FileNotFoundError(f"Route file missing: {ROUTES_XML}")
if not SUMO_CFG.exists():
    raise FileNotFoundError(f"SUMO config missing: {SUMO_CFG}")


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


def write_log(run_id: str, vehicle_id: str, soc_pct: float, station_id: str, score: float, travel_time_min: float, queue_wait_min: float, charge_time_min: float, chosen: int) -> None:
    with sqlite3.connect(DB) as conn:
        conn.execute(
            "INSERT INTO recommendation_log (run_id, vehicle_id, timestamp, soc_pct_at_decision, station_id, score, travel_time_min, queue_wait_min, charge_time_min, chosen) VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?)",
            (run_id, vehicle_id, soc_pct, station_id, score, travel_time_min, queue_wait_min, charge_time_min, chosen),
        )
        conn.commit()


def ensure_station_database(lookup: dict[str, Any], db_path: Path) -> None:
    created = False
    if not db_path.exists():
        create_db(db_path)
        created = True

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT station_id FROM stations")
    existing_stations = {row[0] for row in cursor.fetchall()}

    for station in lookup.values():
        station_id = station["station_id"]
        if station_id not in existing_stations:
            cursor.execute(
                "INSERT OR REPLACE INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    station_id,
                    station.get("name", station_id),
                    float(station.get("lat", 0.0)),
                    float(station.get("lon", 0.0)),
                    station.get("operator", "SUMO"),
                    station.get("address", "Generated station"),
                    f"sim_zone_{station_id}",
                    "simulated",
                ),
            )
            total_ports = int(station.get("num_ports", 1))
            for idx in range(1, total_ports + 1):
                port_id = f"{station_id}_p{idx}"
                cursor.execute(
                    "INSERT OR REPLACE INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        port_id,
                        station_id,
                        "Type2",
                        float(station.get("power_kw", 22.0)),
                        "Available",
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

    conn.commit()
    conn.close()


def estimate_battery_need(vehicle_id: str, soc_pct: float, manifest: dict[str, dict[str, Any]]) -> float:
    details = manifest.get(vehicle_id, {})
    battery_kwh = float(details.get("battery_kwh", 40.0))
    battery_pct = max(0.0, min(soc_pct, 100.0))
    return round(max(0.0, battery_kwh * (1.0 - battery_pct / 100.0)), 2)

def station_available_ports(simulator: OccupancySimulator, station_id: str) -> int:
    return sum(1 for port_state in simulator.port_states.values() if port_state.port_id.startswith(f"{station_id}_") and port_state.is_available())


def route_vehicle_to_station(vehicle_id: str, station_id: str, energy_kwh: float, power_kw: float) -> bool:
    duration_sec = max(300, int((energy_kwh / max(1.0, power_kw)) * 3600))
    try:
        traci.vehicle.setChargingStationStop(vehicle_id, station_id, duration=duration_sec)
        traci.vehicle.rerouteTraveltime(vehicle_id)
        return True
    except Exception as exc:
        print(f"Failed to route {vehicle_id} to {station_id}: {exc}")
        return False


def get_vehicle_soc(vehicle_id: str, manifest: dict[str, dict[str, Any]]) -> float:
    details = manifest.get(vehicle_id, {})
    if details:
        return float(details.get("starting_soc_pct", 100.0))
    try:
        soc = float(traci.vehicle.getParameter(vehicle_id, "device.battery.actualBatteryCapacity"))
        max_soc = float(traci.vehicle.getParameter(vehicle_id, "device.battery.maximumBatteryCapacity"))
        return (soc / max_soc) * 100.0 if max_soc else 100.0
    except Exception:
        return 100.0


def main() -> None:
    init_db()
    run_id = "phase2_run"
    stations_lookup = json.loads(LOOKUP.read_text(encoding="utf-8")) if LOOKUP.exists() else {}
    fleet_manifest = json.loads(FLEET_MANIFEST.read_text(encoding="utf-8")) if FLEET_MANIFEST.exists() else []
    manifest = {entry["vehicle_id"]: entry for entry in fleet_manifest}
    manifest_ids = set(manifest.keys())

    ensure_station_database(stations_lookup, STATION_DB)
    occupancy_simulator = OccupancySimulator(STATION_DB)
    pricing_manager = PricingManager(STATION_DB)
    pricing_manager.update_prices()

    sumo_binary = checkBinary("sumo")
    traci.start([sumo_binary, "-c", str(SUMO_CFG), "--no-step-log", "true", "--verbose", "false"])

    try:
        already_routed: set[str] = set()
        station_vehicle_ids: dict[str, set[str]] = defaultdict(set)

        for step in range(300):
            traci.simulationStep()

            for station_id, station_data in stations_lookup.items():
                try:
                    vehicle_ids = set(traci.chargingstation.getVehicleIDs(station_id))
                except Exception:
                    vehicle_ids = set()

                arrivals = vehicle_ids - station_vehicle_ids[station_id]
                for vehicle in arrivals:
                    soc_pct = get_vehicle_soc(vehicle, manifest)
                    battery_need_kwh = estimate_battery_need(vehicle, soc_pct, manifest)
                    port_id = occupancy_simulator.process_arrival(station_id, battery_need_kwh)
                    if port_id == "no_available_port":
                        print(f"Step {step}: {vehicle} arrived at {station_id}, no available port")
                    else:
                        print(f"Step {step}: {vehicle} assigned to {port_id} at {station_id}")
                station_vehicle_ids[station_id] = vehicle_ids

            if step % 50 == 0:
                pricing_manager.update_prices()

            occupancy_simulator.tick()

            active_vehicles = traci.vehicle.getIDList()
            if not active_vehicles:
                print("No active vehicles in SUMO simulation")

            for vehicle in active_vehicles:
                if manifest_ids and vehicle not in manifest_ids:
                    continue
                if vehicle in already_routed:
                    continue

                soc_pct = get_vehicle_soc(vehicle, manifest)
                if soc_pct < 80.0:
                    print(f"Vehicle {vehicle} SoC {soc_pct:.1f}%")
                    candidates: list[dict[str, Any]] = []
                    for station_id, station_data in stations_lookup.items():
                        total_ports = int(station_data.get("num_ports", 1))
                        free_ports = station_available_ports(occupancy_simulator, station_id)
                        candidates.append(
                            {
                                "station_id": station_id,
                                "distance_km": float(station_data.get("distance_km", 2.0)),
                                "travel_time_min": float(station_data.get("travel_time_min", 4.0)),
                                "free_ports": free_ports,
                                "total_ports": total_ports,
                                "power_kw": float(station_data.get("power_kw", 22.0)),
                            }
                        )

                    engine = RecommendationEngine()
                    scored = [
                        engine.score_station(
                            {"soc_pct": soc_pct, "battery_kwh": manifest.get(vehicle, {}).get("battery_kwh", 40.0), "consumption_wh_per_km": manifest.get(vehicle, {}).get("consumption_wh_per_km", 180.0)},
                            StationCandidate(
                                station_id=candidate["station_id"],
                                distance_km=candidate["distance_km"],
                                travel_time_min=candidate["travel_time_min"],
                                free_ports=candidate["free_ports"],
                                total_ports=candidate["total_ports"],
                                power_kw=candidate["power_kw"],
                            ),
                        )
                        for candidate in candidates
                    ]
                    scored.sort(key=lambda item: item.score)
                    if scored:
                        best = scored[0]
                        already_routed.add(vehicle)
                        write_log(run_id, vehicle, soc_pct, best.station_id, best.score, best.travel_time_min, best.queue_wait_min, best.charge_time_min, 1)
                        print(f"Recommended {vehicle} to {best.station_id} at SoC {soc_pct:.1f}%")
                        route_vehicle_to_station(vehicle, best.station_id, estimate_battery_need(vehicle, soc_pct, manifest), float(stations_lookup[best.station_id].get("power_kw", 22.0)))

            if step % 50 == 0:
                print(f"step {step}: active vehicles={len(active_vehicles)}")

        print("Simulation complete")
    finally:
        occupancy_simulator.close()
        traci.close()


if __name__ == "__main__":
    main()
