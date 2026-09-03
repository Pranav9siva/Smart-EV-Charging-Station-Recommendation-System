"""Compare PPO with simple station-selection baselines under identical SUMO conditions.

Policies:
- ppo: existing production recommendation path
- nearest: minimum station distance
- cheapest: minimum price per kWh
- minimum_wait: minimum estimated waiting time

This harness patches only the evaluator's recommendation call for baseline
runs. Production controller and PPO code are not changed.

Example:
    python scripts/compare_ppo_baselines.py --episodes 2 --steps 1200
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from statistics import mean
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import traci
import sumolib

from src.ev_model.battery import energy_needed_kwh
from src.simulation.controller import SimulationController

DB_PATH = ROOT / "data" / "stations" / "stations.sqlite"
MODEL_PATH = ROOT / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"
SUMO_CFG = ROOT / "simulations" / "bangalore" / "sim.sumocfg"
OUTPUT_DIR = ROOT / "runs" / "evaluation" / "ppo_baseline_comparison"
POLICIES = ("ppo", "nearest", "cheapest", "minimum_wait")
METRICS = ("reward", "charging_cost", "waiting_time_min", "travel_distance_km", "successful_charging")
_NETWORK_CACHE: Any = None


def _shared_network() -> Any:
    global _NETWORK_CACHE
    if _NETWORK_CACHE is None:
        _NETWORK_CACHE = sumolib.net.readNet(str(ROOT / "simulations" / "bangalore" / "network.net.xml"))
    return _NETWORK_CACHE


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    value = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return radius * 2.0 * math.atan2(math.sqrt(value), math.sqrt(max(1e-12, 1.0 - value)))


def _candidate_rows(manager: Any, vehicle_state: dict[str, Any]) -> list[dict[str, Any]]:
    lat = float(vehicle_state.get("lat", 12.9716) or 12.9716)
    lon = float(vehicle_state.get("lon", 77.5946) or 77.5946)
    rows: list[dict[str, Any]] = []
    for station in manager.list_all_stations():
        station_id = str(station.get("station_id") or "")
        if not station_id:
            continue
        metrics = manager.get_station_metrics(station_id) or {}
        station_lat = station.get("lat") if station.get("lat") is not None else metrics.get("lat")
        station_lon = station.get("lon") if station.get("lon") is not None else metrics.get("lon")
        if station_lat is None or station_lon is None:
            continue
        distance = _haversine_km(lat, lon, float(station_lat), float(station_lon))
        available_ports = int(metrics.get("available_ports", 0) or 0)
        total_ports = max(1, int(metrics.get("total_ports", 1) or 1))
        price = float(metrics.get("price_per_kwh", 0.0) or 0.0)
        wait = float(metrics.get("avg_wait_estimate", 0.0) or 0.0)
        queue = int(metrics.get("queue_length", max(0, total_ports - available_ports)) or 0)
        power = float(metrics.get("power_kw", metrics.get("grid_load_kw", 0.0)) or 0.0)
        rows.append({
            "station_id": station_id,
            "distance_km": distance,
            "travel_time_min": max(0.1, distance / 24.0 * 60.0),
            "available_ports": available_ports,
            "total_ports": total_ports,
            "price_per_kwh": price,
            "waiting_time": wait,
            "queue_length": queue,
            "grid_load": power,
        })
    return rows


def _baseline_recommendation(manager: Any, vehicle_state: dict[str, Any], policy: str) -> dict[str, Any]:
    candidates = _candidate_rows(manager, vehicle_state)
    if not candidates:
        return {"station": "", "selected_station": "", "recommendation_score": 0.0, "recommendation_reason": f"baseline:{policy}:no_candidate"}
    if policy == "nearest":
        chosen = min(candidates, key=lambda row: (row["distance_km"], row["station_id"]))
        reason = "baseline nearest station"
    elif policy == "cheapest":
        chosen = min(candidates, key=lambda row: (row["price_per_kwh"], row["distance_km"], row["station_id"]))
        reason = "baseline cheapest station"
    elif policy == "minimum_wait":
        chosen = min(candidates, key=lambda row: (row["waiting_time"], row["queue_length"], row["distance_km"], row["station_id"]))
        reason = "baseline minimum waiting time"
    else:
        raise ValueError(f"Unknown baseline policy: {policy}")

    battery_pct = float(vehicle_state.get("battery_pct", 0.0) or 0.0)
    battery_kwh = float(vehicle_state.get("battery_capacity_kwh", 0.0) or 0.0)
    energy_need = energy_needed_kwh(battery_pct, 100.0, battery_kwh)
    reward = -chosen["waiting_time"] * 0.5 - chosen["distance_km"] * 0.2
    reward -= energy_need * chosen["price_per_kwh"] * 0.1
    reward += chosen["available_ports"] * 0.5 - chosen["queue_length"] - chosen["grid_load"] * 0.01
    if float(vehicle_state.get("remaining_range_km", 0.0) or 0.0) < chosen["distance_km"] * 1.2:
        reward -= 50.0
    return {
        "station": chosen["station_id"],
        "selected_station": chosen["station_id"],
        "station_name": chosen["station_id"],
        "travel_distance": chosen["distance_km"],
        "travel_time": chosen["travel_time_min"],
        "waiting_time": chosen["waiting_time"],
        "queue_length": chosen["queue_length"],
        "charging_cost": energy_need * chosen["price_per_kwh"],
        "available_ports": chosen["available_ports"],
        "grid_load": chosen["grid_load"],
        "recommendation_score": reward,
        "ppo_reward": None,
        "recommendation_reason": reason,
    }


def _run_policy_episode(policy: str, episode: int, steps: int, fleet_size: int, tracked: int, seed: int) -> dict[str, Any]:
    try:
        traci.close()
    except Exception:
        pass
    controller = SimulationController(
        sumo_cfg=str(SUMO_CFG), db_path=str(DB_PATH), model_path=str(MODEL_PATH),
        fleet_size=fleet_size, station_count=500, tracked=tracked, charge_threshold_pct=20.0,
        use_gui=False, dashboard_state_interval=steps + 1, dashboard_report_interval=steps + 1,
        visualization_interval=steps + 1, metrics_interval=steps + 1, simulation_seed=seed,
    )
    controller._refresh_dashboard = lambda: None
    controller._collect_metrics = lambda _step: None
    started = time.perf_counter()
    steps_completed = 0
    original_recommend = None
    try:
        controller.start()
        # Reuse one parsed Bengaluru network across policy episodes. This is an
        # evaluation-only performance optimization; SUMO and controller state
        # remain fresh for every policy/episode.
        controller.net = _shared_network()
        for vehicle in controller.vehicle_manager.list_tracked_vehicles():
            vehicle.battery_pct = 15.0
            vehicle.remaining_range_km = max(1.0, vehicle.remaining_range_km * 0.15)

        if policy != "ppo":
            original_recommend = controller.recommender.recommend

            def recommend_baseline(vehicle_state: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
                return _baseline_recommendation(controller.station_manager, vehicle_state, policy)

            controller.recommender.recommend = recommend_baseline  # type: ignore[method-assign]

        for step in range(steps):
            traci.simulationStep()
            try:
                controller.current_step = int(traci.simulation.getTime())
            except Exception:
                controller.current_step = step
            controller._update_vehicles(step)
            controller._process_charging(step)
            controller._invalidate_station_snapshots()
            steps_completed += 1

        recommendations = list(controller.recommendation_log)
        events = list(controller.charging_events)
        rewards = [float(item.get("ppo_reward") if item.get("ppo_reward") is not None else item.get("recommendation_score", 0.0) or 0.0) for item in recommendations]
        waits = [float(item.get("waiting_time", 0.0) or 0.0) for item in recommendations]
        distances = [float(item.get("distance_km", 0.0) or 0.0) for item in recommendations]
        completed_events = [event for event in events if event.get("event_type") == "charging_completed"]
        return {
            "policy": policy,
            "episode": episode,
            "seed": seed,
            "steps_requested": steps,
            "steps_completed": steps_completed,
            "recommendations": len(recommendations),
            "reward": sum(rewards),
            "charging_cost": sum(float(event.get("session_charging_cost", 0.0) or 0.0) for event in completed_events),
            "waiting_time_min": sum(waits),
            "travel_distance_km": sum(distances),
            "successful_charging": len(completed_events),
            "charging_started": sum(1 for event in events if event.get("event_type") == "charging_started"),
            "charging_completed": len(completed_events),
            "status": "PASS" if steps_completed == steps else "INCOMPLETE",
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }
    finally:
        if original_recommend is not None:
            controller.recommender.recommend = original_recommend  # type: ignore[method-assign]
        controller.stop()
        try:
            traci.close()
        except Exception:
            pass


def _run_with_retry(policy: str, episode: int, steps: int, fleet_size: int, tracked: int, seed: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            return _run_policy_episode(policy, episode, steps, fleet_size, tracked, seed)
        except Exception as exc:
            last_error = exc
            try:
                traci.close()
            except Exception:
                pass
            time.sleep(0.5)
    return {
        "policy": policy,
        "episode": episode,
        "seed": seed,
        "steps_requested": steps,
        "steps_completed": 0,
        "recommendations": 0,
        "reward": 0.0,
        "charging_cost": 0.0,
        "waiting_time_min": 0.0,
        "travel_distance_km": 0.0,
        "successful_charging": 0,
        "charging_started": 0,
        "charging_completed": 0,
        "runtime_seconds": 0.0,
        "status": "ERROR",
        "error": repr(last_error),
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {"average": round(mean(values), 4) if values else 0.0, "minimum": round(min(values), 4) if values else 0.0, "maximum": round(max(values), 4) if values else 0.0}


def _write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    for policy in POLICIES:
        policy_rows = [row for row in rows if row["policy"] == policy]
        aggregates[policy] = {metric: _stats([float(row.get(metric, 0.0) or 0.0) for row in policy_rows]) for metric in METRICS}
    result = {
        "evaluation": "ppo_baseline_comparison",
        "schema_version": 1,
        "conditions": {"episodes": args.episodes, "steps": args.steps, "fleet_size": args.fleet_size, "tracked": args.tracked, "seed_base": args.seed},
        "policies": list(POLICIES),
        "aggregates": aggregates,
        "episodes": rows,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "ppo_baseline_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = ["policy", "episode", "seed", "steps_requested", "steps_completed", "recommendations", *METRICS, "charging_started", "charging_completed", "runtime_seconds", "status"]
    with (OUTPUT_DIR / "ppo_baseline_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 3, figsize=(15, 8))
        for axis, metric in zip(axes.flat, METRICS):
            values = [aggregates[policy][metric]["average"] for policy in POLICIES]
            axis.bar(list(POLICIES), values, color=["#2563eb", "#64748b", "#f59e0b", "#22c55e"])
            axis.set_title(metric.replace("_", " ").title())
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.25)
        axes.flat[-1].axis("off")
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / "ppo_baseline_comparison.png", dpi=150)
        plt.close(figure)
        result["plot"] = "ppo_baseline_comparison.png"
    except Exception as exc:
        result["plot"] = None
        result["plot_error"] = str(exc)
    (OUTPUT_DIR / "ppo_baseline_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--fleet-size", type=int, default=10)
    parser.add_argument("--tracked", type=int, default=10)
    parser.add_argument("--seed", type=int, default=5200)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for episode in range(1, args.episodes + 1):
            row = _run_with_retry(policy, episode, args.steps, args.fleet_size, args.tracked, args.seed + episode - 1)
            rows.append(row)
            print(f"policy={policy} episode={episode} steps={row['steps_completed']}/{args.steps} recommendations={row['recommendations']} successful_charging={row['successful_charging']}")
    _write_outputs(rows, args)
    print(f"Results written to {OUTPUT_DIR}")
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
