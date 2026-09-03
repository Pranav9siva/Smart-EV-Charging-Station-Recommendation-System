from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.ev_model.battery import energy_needed_kwh
from src.rl_env.gym_ev_charging_env import GymEVChargingEnv

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "stations" / "stations.sqlite"
OUTPUT_DIR = ROOT / "runs" / "evaluation" / "behavioral_eval"


class BehavioralEvaluationEnv(GymEVChargingEnv):
    """A behavioral extension of GymEVChargingEnv that records actual charge state transitions."""

    def __init__(self, db_path: str | None = None, tracked_vehicle_count: int = 10, candidate_count: int = 8) -> None:
        super().__init__(db_path=db_path, tracked_vehicle_count=tracked_vehicle_count, candidate_count=candidate_count)

    def step(self, action):
        chosen_idx = int(action) if action is not None else 0
        chosen_idx = max(0, min(chosen_idx, self.candidate_count - 1))
        chosen = self.current_candidates[chosen_idx]

        vehicles = self.vehicle_manager.list_tracked_vehicles()
        vehicle = vehicles[0] if vehicles else None
        soc_before = float(vehicle.battery_pct) if vehicle else 0.0
        range_before = float(vehicle.remaining_range_km) if vehicle else 0.0

        charge_needed = bool(vehicle is not None and vehicle.battery_pct <= 20.0)
        charge_event = bool(charge_needed and chosen.get("station_id") and int(chosen.get("free_ports", 0)) > 0)
        successful_charging_decision = bool(charge_event and chosen.get("station_id"))
        if successful_charging_decision and vehicle is not None:
            energy_needed = energy_needed_kwh(vehicle.battery_pct, 80.0, vehicle.battery_capacity_kwh)
            charge_gain_pct = min(20.0, max(0.0, energy_needed / max(1.0, vehicle.battery_capacity_kwh) * 100.0))
            vehicle.battery_pct = min(100.0, vehicle.battery_pct + charge_gain_pct)
            vehicle.remaining_range_km = max(vehicle.remaining_range_km, vehicle.battery_capacity_kwh * 1000.0 / 200.0 * (vehicle.battery_pct / 100.0))

        soc_after = float(vehicle.battery_pct) if vehicle else 0.0
        range_after = float(vehicle.remaining_range_km) if vehicle else 0.0

        reward = 0.0
        for v in vehicles:
            dist = chosen.get("distance_km", 1e6)
            if v.remaining_range_km < dist * 1.2:
                reward -= 50.0
            reward -= float(chosen.get("avg_wait_min", 0.0)) * 0.5
            reward -= float(dist) * 0.2
            energy_need = energy_needed_kwh(v.battery_pct, 100.0, v.battery_capacity_kwh)
            reward -= energy_need * float(chosen.get("price_per_kwh", 0.0)) * 0.1
            reward += float(chosen.get("free_ports", 0)) * 0.5
            reward -= float(chosen.get("queue_len", 0)) * 1.0
            reward -= float(chosen.get("grid_load_kw", 0.0)) * 0.01

        self.episode_length += 1
        self.episode_reward += float(reward)
        obs = self._build_observation()
        done = False
        info = {
            "chosen_station": chosen.get("station_id", ""),
            "behavior": {
                "charge_needed": charge_needed,
                "charging_event": charge_event,
                "successful_charging_decision": successful_charging_decision,
                "soc_before": soc_before,
                "soc_after": soc_after,
                "range_before": range_before,
                "range_after": range_after,
                "selected_station_id": chosen.get("station_id", ""),
                "free_ports": int(chosen.get("free_ports", 0)),
                "queue_length": int(chosen.get("queue_len", 0)),
            },
        }
        return obs, float(reward), done, False, info


def run_behavioral_eval() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = BehavioralEvaluationEnv(db_path=str(DB_PATH), tracked_vehicle_count=1, candidate_count=1)
    try:
        obs, _ = env.reset(seed=7)
        vehicle = env.vehicle_manager.list_tracked_vehicles()[0]
        vehicle.battery_pct = 10.0
        vehicle.remaining_range_km = 1.0
        env.current_candidates = [
            {
                "station_id": "sim_station_1",
                "distance_km": 2.0,
                "travel_time_min": 5.0,
                "free_ports": 1,
                "total_ports": 2,
                "price_per_kwh": 0.2,
                "avg_wait_min": 0.0,
                "queue_len": 0,
                "grid_load_kw": 10.0,
            }
        ]
        obs, reward, terminated, truncated, info = env.step(0)
        payload = {
            "reward": reward,
            "behavior": info["behavior"],
            "observation_shape": {
                "vehicles": obs["vehicles"].shape,
                "stations": obs["stations"].shape,
            },
        }
        with (OUTPUT_DIR / "behavioral_eval.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload
    finally:
        env.close()


if __name__ == "__main__":
    run_behavioral_eval()
