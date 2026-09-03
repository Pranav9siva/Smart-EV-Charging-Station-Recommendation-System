from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import traci

from scripts.generate_ev_fleet import build_routes
from src.ev_model.battery import energy_needed_kwh
from src.simulation.controller import SimulationController

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "simulations" / "bangalore" / "sim.sumocfg"
DB = ROOT / "data" / "stations" / "stations.sqlite"
MODEL = ROOT / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"
OUT = ROOT / "outputs" / "refresh_dashboard_profile.json"


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _profile_refresh_dashboard_shadow(ctrl: SimulationController, *, step: int, include_visualization_publish: bool) -> dict[str, Any]:
    stage_ms: dict[str, float] = {
        "station_db_queries_ms": 0.0,
        "tracked_vehicle_processing_ms": 0.0,
        "recommendation_metrics_collection_ms": 0.0,
        "dashboard_state_construction_ms": 0.0,
        "json_file_writing_ms": 0.0,
        "visualization_publish_ms": 0.0,
    }

    vehicles: list[dict[str, Any]] = []
    stations: list[dict[str, Any]] = []

    t0 = _now_ms()
    station_rows = ctrl.station_manager.list_all_stations()
    for row in station_rows:
        sid = row["station_id"]
        metrics = ctrl.station_manager.get_station_metrics(sid) or {}
        stations.append(
            {
                "station_id": sid,
                "price_per_kwh": metrics.get("price_per_kwh"),
                "avg_wait_min": metrics.get("avg_wait_estimate"),
                "available_ports": metrics.get("available_ports"),
                "grid_load_kw": metrics.get("power_kw", 0.0),
                "queue_len": max(0, metrics.get("total_ports", 0) - metrics.get("available_ports", 0)),
            }
        )
    stage_ms["station_db_queries_ms"] = _now_ms() - t0

    for v in ctrl.vehicle_manager.list_vehicles():
        vdict = v.to_dict()
        if v.tracked:
            t_stage2 = _now_ms()
            sumo_vid = ctrl.vm_to_sumo.get(v.vehicle_id, v.vehicle_id)
            try:
                current_ids = set(traci.vehicle.getIDList())
            except Exception:
                current_ids = set()

            if sumo_vid not in current_ids:
                edge = None
                pos = (None, None)
            else:
                try:
                    edge = traci.vehicle.getRoadID(sumo_vid)
                except Exception:
                    edge = None
                try:
                    pos = traci.vehicle.getPosition(sumo_vid)
                except Exception:
                    pos = (None, None)

            assigned = ctrl.assignments.get(v.vehicle_id)
            rec_station = assigned.station_id if assigned else None
            charging_status = assigned.status if assigned else "driving"

            distance = None
            travel_time = None
            charging_price_per_kwh = None
            estimated_session_cost = None
            session_charging_cost = None
            queue_len = None
            waiting_time = None
            available_ports = None
            occupied_ports = None
            grid_load = None

            if rec_station:
                st = ctrl.station_manager.get_station_state(rec_station)
                metrics = ctrl.station_manager.get_station_metrics(rec_station) or {}
                if assigned is not None:
                    travel_time = assigned.recommended_travel_time_min
                    waiting_time = assigned.recommended_waiting_time_min
                    queue_len = assigned.recommended_queue_length
                    charging_price_per_kwh = assigned.charging_price_per_kwh
                    estimated_session_cost = assigned.estimated_session_cost
                    session_charging_cost = assigned.session_charging_cost
                if st:
                    try:
                        lat = st.get("lat")
                        lon = st.get("lon")
                        if lat is not None and lon is not None and pos and pos[0] is not None and pos[1] is not None and ctrl.net is not None:
                            sx, sy = ctrl.net.convertLonLat2XY(float(lon), float(lat))
                            distance = round(math.hypot(float(pos[0]) - float(sx), float(pos[1]) - float(sy)) / 1000.0, 3)
                    except Exception:
                        distance = None

                if charging_price_per_kwh is None and metrics.get("price_per_kwh") is not None:
                    charging_price_per_kwh = float(metrics.get("price_per_kwh"))
                if queue_len is None:
                    queue_len = max(0, metrics.get("total_ports", 0) - metrics.get("available_ports", 0))
                if waiting_time is None:
                    waiting_time = metrics.get("avg_wait_estimate")
                available_ports = metrics.get("available_ports")
                occupied_ports = metrics.get("occupied_ports")
                st_state = ctrl.station_manager.get_station_state(rec_station) or {}
                ports = st_state.get("ports", [])
                grid_load = sum(float(p.get("power_kw", 0.0)) for p in ports if p.get("status") == "Charging")

            if estimated_session_cost is None and charging_price_per_kwh is not None:
                estimated_session_cost = max(0.0, energy_needed_kwh(v.battery_pct, 80.0, v.battery_capacity_kwh)) * charging_price_per_kwh
            if session_charging_cost is None:
                session_charging_cost = 0.0
            stage_ms["tracked_vehicle_processing_ms"] += _now_ms() - t_stage2

            t_stage3 = _now_ms()
            ppo_reward = None
            for r in reversed(ctrl.recommendation_log):
                if r.get("vehicle_id") == v.vehicle_id:
                    ppo_reward = r.get("ppo_reward")
                    break
            stage_ms["recommendation_metrics_collection_ms"] += _now_ms() - t_stage3

            vdict.update(
                {
                    "id": v.vehicle_id,
                    "current_edge": edge,
                    "current_position": {"x": pos[0], "y": pos[1]} if pos else None,
                    "recommended_station": rec_station,
                    "distance": distance,
                    "travel_time": travel_time,
                    "charging_price_per_kwh": charging_price_per_kwh,
                    "estimated_session_cost": estimated_session_cost,
                    "session_charging_cost": session_charging_cost,
                    "charging_cost": session_charging_cost,
                    "queue_length": queue_len,
                    "waiting_time": waiting_time,
                    "available_ports": available_ports,
                    "occupied_ports": occupied_ports,
                    "grid_load": grid_load,
                    "charging_status": charging_status,
                    "ppo_reward": ppo_reward,
                    "simulation_step": getattr(ctrl, "current_step", None),
                }
            )

        vehicles.append(vdict)

    t4 = _now_ms()
    tracked_ids = [v.vehicle_id for v in ctrl.vehicle_manager.list_tracked_vehicles()]
    wait_values = [station.get("avg_wait_min", 0.0) for station in stations if station.get("avg_wait_min") is not None]
    summary = {
        "charging_vehicles": sum(1 for vehicle in vehicles if str(vehicle.get("charging_status") or "").lower() == "charging"),
        "avg_wait_min": round(statistics.mean(wait_values) if wait_values else 0.0, 2),
        "active_assignments": len(ctrl.assignments),
        "total_vehicles": len(vehicles),
    }
    state_payload = {
        "vehicles": vehicles,
        "stations": stations,
        "tracked_ids": tracked_ids,
        "summary": summary,
        "metrics": ctrl.simulation_metrics[-20:],
    }
    stage_ms["dashboard_state_construction_ms"] = _now_ms() - t4

    t5 = _now_ms()
    ctrl.dashboard.update(state_payload)
    stage_ms["json_file_writing_ms"] = _now_ms() - t5

    if include_visualization_publish and getattr(ctrl, "visualization", None) is not None:
        t6 = _now_ms()
        ctrl.visualization.publish_step(step)
        stage_ms["visualization_publish_ms"] = _now_ms() - t6

    stage_ms["refresh_total_ms"] = sum(stage_ms.values())
    return stage_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Use SUMO-GUI")
    parser.add_argument("--steps", type=int, default=3, help="Number of simulation steps to profile")
    parser.add_argument("--include-visualization-publish", action="store_true", help="Also profile visualization publish call per step")
    args = parser.parse_args()

    build_routes(vehicle_count=1000, tracked_count=10, depart_window=120, seed=42)

    ctrl = SimulationController(
        sumo_cfg=str(CFG),
        db_path=str(DB),
        model_path=str(MODEL),
        fleet_size=1000,
        station_count=0,
        tracked=10,
        use_gui=bool(args.gui),
    )

    profile_rows: list[dict[str, Any]] = []
    ctrl_start_ms = None
    try:
        t0 = _now_ms()
        ctrl.start()
        ctrl_start_ms = _now_ms() - t0

        for step in range(int(args.steps)):
            traci.simulationStep()
            try:
                ctrl.current_step = int(traci.simulation.getTime())
            except Exception:
                ctrl.current_step = step
            ctrl._update_vehicles(step)
            # Keep charging state transitions active so dashboard values are realistic.
            ctrl._process_charging(step)
            stage = _profile_refresh_dashboard_shadow(
                ctrl,
                step=step,
                include_visualization_publish=bool(args.include_visualization_publish),
            )
            stage["step"] = step
            profile_rows.append(stage)

    finally:
        ctrl.stop()

    keys = [
        "station_db_queries_ms",
        "tracked_vehicle_processing_ms",
        "recommendation_metrics_collection_ms",
        "dashboard_state_construction_ms",
        "json_file_writing_ms",
        "visualization_publish_ms",
        "refresh_total_ms",
    ]
    aggregates = {
        f"avg_{key}": round(statistics.mean([row[key] for row in profile_rows]), 3) if profile_rows else 0.0
        for key in keys
    }
    aggregates.update(
        {
            f"max_{key}": round(max([row[key] for row in profile_rows]), 3) if profile_rows else 0.0
            for key in keys
        }
    )

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": int(args.steps),
        "use_gui": bool(args.gui),
        "include_visualization_publish": bool(args.include_visualization_publish),
        "ctrl_start_ms": round(ctrl_start_ms or 0.0, 3),
        "rows": profile_rows,
        "aggregates": aggregates,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Wrote refresh profile to {OUT}")
    print(json.dumps(result["aggregates"], indent=2))


if __name__ == "__main__":
    main()
