"""Evaluate the production PPO recommendation path over multiple SUMO episodes.

This is an evaluation harness. It does not modify the production controller or
recommendation algorithm; it drives the same controller stages with dashboard
and metrics publishing disabled to keep the benchmark bounded.

Example:
    python scripts/evaluate_ppo_episodes.py --episodes 2 --steps 1200
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import traci

from src.simulation.controller import SimulationController

DB_PATH = ROOT / "data" / "stations" / "stations.sqlite"
MODEL_PATH = ROOT / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"
SUMO_CFG = ROOT / "simulations" / "bangalore" / "sim.sumocfg"
OUTPUT_DIR = ROOT / "runs" / "evaluation" / "ppo_multi_episode"

METRIC_NAMES = (
    "reward",
    "charging_cost",
    "waiting_time_min",
    "travel_distance_km",
    "successful_charging",
)


def _mean_min_max(values: list[float]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"average": 0.0, "minimum": 0.0, "maximum": 0.0}
    return {
        "average": round(mean(numbers), 4),
        "minimum": round(min(numbers), 4),
        "maximum": round(max(numbers), 4),
    }


def _event_count(events: list[dict[str, Any]], event_type: str) -> int:
    return sum(1 for event in events if event.get("event_type") == event_type)


def _run_episode(episode: int, steps: int, fleet_size: int, tracked: int, seed: int) -> dict[str, Any]:
    controller = SimulationController(
        sumo_cfg=str(SUMO_CFG),
        db_path=str(DB_PATH),
        model_path=str(MODEL_PATH),
        fleet_size=fleet_size,
        station_count=500,
        tracked=tracked,
        charge_threshold_pct=20.0,
        use_gui=False,
        dashboard_state_interval=steps + 1,
        dashboard_report_interval=steps + 1,
        visualization_interval=steps + 1,
        metrics_interval=steps + 1,
        simulation_seed=seed,
    )
    controller._refresh_dashboard = lambda: None
    controller._collect_metrics = lambda _step: None
    started = time.perf_counter()
    steps_completed = 0
    try:
        controller.start()

        # Make every tracked EV exercise the existing low-SOC recommendation path.
        for vehicle in controller.vehicle_manager.list_tracked_vehicles():
            vehicle.battery_pct = 15.0
            vehicle.remaining_range_km = max(1.0, vehicle.remaining_range_km * 0.15)

        for step in range(max(0, int(steps))):
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
        charging_events = list(controller.charging_events)
        reward_values = [
            float(item.get("ppo_reward"))
            if item.get("ppo_reward") is not None
            else float(item.get("recommendation_score", 0.0) or 0.0)
            for item in recommendations
        ]
        estimated_costs = [float(item.get("estimated_session_cost", 0.0) or 0.0) for item in recommendations]
        waits = [float(item.get("waiting_time", 0.0) or 0.0) for item in recommendations]
        distances = [float(item.get("distance_km", 0.0) or 0.0) for item in recommendations]
        completed = _event_count(charging_events, "charging_completed")
        realized_cost = sum(
            float(event.get("session_charging_cost", 0.0) or 0.0)
            for event in charging_events
            if event.get("event_type") == "charging_completed"
        )

        return {
            "episode": episode,
            "seed": seed,
            "steps_requested": int(steps),
            "steps_completed": steps_completed,
            "recommendations": len(recommendations),
            "reward": sum(reward_values),
            "charging_cost": realized_cost,
            "estimated_charging_cost": sum(estimated_costs),
            "waiting_time_min": sum(waits),
            "travel_distance_km": sum(distances),
            "successful_charging": completed,
            "charging_started": _event_count(charging_events, "charging_started"),
            "charging_completed": completed,
            "queue_events": _event_count(charging_events, "queue_joined"),
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "status": "PASS" if steps_completed == steps else "INCOMPLETE",
        }
    finally:
        controller.stop()


def _write_results(episodes: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    aggregates = {
        metric: _mean_min_max([float(episode.get(metric, 0.0) or 0.0) for episode in episodes])
        for metric in METRIC_NAMES
    }
    result = {
        "evaluation": "ppo_multi_episode",
        "schema_version": 2,
        "policy": "PPO recommendation path",
        "episodes_requested": int(args.episodes),
        "episodes_completed": len(episodes),
        "steps_per_episode": int(args.steps),
        "fleet_size": int(args.fleet_size),
        "tracked": int(args.tracked),
        "aggregates": aggregates,
        "episodes": episodes,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "ppo_multi_episode_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fields = ["episode", "seed", "steps_requested", "steps_completed", "recommendations", *METRIC_NAMES, "estimated_charging_cost", "charging_started", "charging_completed", "queue_events", "runtime_seconds", "status"]
    with (OUTPUT_DIR / "ppo_multi_episode_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episodes)

    with (OUTPUT_DIR / "ppo_multi_episode_aggregates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "average", "minimum", "maximum"])
        writer.writeheader()
        for metric, values in aggregates.items():
            writer.writerow({"metric": metric, **values})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--fleet-size", type=int, default=10)
    parser.add_argument("--tracked", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4200)
    args = parser.parse_args()
    if args.episodes <= 0 or args.steps <= 0:
        parser.error("--episodes and --steps must be positive")

    episodes = []
    for index in range(args.episodes):
        episode = _run_episode(index + 1, args.steps, args.fleet_size, args.tracked, args.seed + index)
        episodes.append(episode)
        print(
            f"episode={episode['episode']} steps={episode['steps_completed']}/{episode['steps_requested']} "
            f"recommendations={episode['recommendations']} completed_charging={episode['successful_charging']}"
        )
    result = _write_results(episodes, args)
    print(json.dumps(result["aggregates"], indent=2))
    print(f"Results written to {OUTPUT_DIR}")
    return 0 if all(episode["status"] == "PASS" for episode in episodes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
