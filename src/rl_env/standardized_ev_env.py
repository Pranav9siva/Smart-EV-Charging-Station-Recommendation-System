"""Standardized Gymnasium Environment for Single-Agent and Multi-Agent RL Comparison.

Provides a unified interface for:
1. DQN (Single-agent discrete action)
2. DDPG (Single-agent continuous action with decoding layer)
3. MADDPG (Multi-agent continuous/discrete with centralized critic)
4. PPO (Single-agent discrete action - baseline)
5. MAPPO (Multi-agent discrete action with centralized critic)
6. Constrained MAPPO (Multi-agent discrete action with explicit dual Lagrangian constraint signals)
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces
from gymnasium.core import Env

from src.station_management.manager import ChargingStationManager
from src.ev_management.vehicle_manager import VehicleManager, VehicleRecord
from src.ev_model.battery import energy_needed_kwh


class StandardizedEVEnv(Env):
    """Standardized EV Charging Station Recommendation Environment."""

    def __init__(
        self,
        db_path: Optional[str] = "data/stations/stations.sqlite",
        station_manager: Optional[ChargingStationManager] = None,
        num_evs: int = 10,
        candidate_count: int = 8,
        scenario: str = "normal",  # normal, high_traffic, high_demand, reduced_ports, high_load, varying_soc
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._owns_station_manager = station_manager is None and bool(db_path)
        self.station_manager = station_manager or (ChargingStationManager(db_path) if db_path else None)
        self.num_evs = int(num_evs)
        self.candidate_count = int(candidate_count)
        self.scenario = scenario
        self._seed = seed

        # EV features: [battery_pct, battery_kwh, remaining_range_km, lat, lon] (dim=5)
        self.ev_feat_dim = 5
        # Station features: [distance_km, travel_time_min, free_ports, total_ports, price_per_kwh, avg_wait_min] (dim=6)
        self.station_feat_dim = 6

        # Single agent observation space: flattened vector of all EVs + candidate stations
        self.obs_dim_per_agent = self.ev_feat_dim + (self.candidate_count * self.station_feat_dim)
        self.global_obs_dim = (self.num_evs * self.ev_feat_dim) + (self.candidate_count * self.station_feat_dim)

        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(self.obs_dim_per_agent,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.candidate_count)

        self.vehicle_manager = VehicleManager()
        self.current_candidates: List[List[Dict[str, Any]]] = []
        self.ev_positions: List[Tuple[float, float]] = []
        self.step_count = 0
        self.max_steps = 100

    def set_scenario(self, scenario: str) -> None:
        self.scenario = scenario

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self._seed = seed

        self.step_count = 0
        self.vehicle_manager = VehicleManager()
        self.vehicle_manager.generate_fleet(count=max(100, self.num_evs * 2), detail_count=self.num_evs)
        tracked = self.vehicle_manager.list_tracked_vehicles()

        # Adjust initial SOC based on scenario
        self.ev_positions = []
        for i, vrec in enumerate(tracked):
            if self.scenario == "high_demand":
                vrec.battery_pct = float(np.random.uniform(5.0, 20.0))
            elif self.scenario == "varying_soc":
                vrec.battery_pct = float(np.random.uniform(2.0, 95.0))
            else:
                vrec.battery_pct = float(np.random.uniform(10.0, 35.0))
            
            vrec.remaining_range_km = (vrec.battery_pct / 100.0) * (vrec.battery_capacity_kwh / 0.20)
            
            # Sample positions around Bengaluru
            lat = 12.9716 + np.random.uniform(-0.08, 0.08)
            lon = 77.5946 + np.random.uniform(-0.08, 0.08)
            self.ev_positions.append((lat, lon))

        self._sample_candidates_all_evs()
        obs = self.get_single_agent_obs(0)
        return obs, {}

    def _sample_candidates_all_evs(self) -> None:
        self.current_candidates = []
        all_stations = []
        if self.station_manager:
            try:
                all_stations = list(self.station_manager.list_all_stations())
            except Exception:
                all_stations = []

        if not all_stations:
            all_stations = [
                {
                    "station_id": f"station_{i}",
                    "lat": 12.9716 + (i * 0.01),
                    "lon": 77.5946 + (i * 0.01),
                    "available_ports": 2 + (i % 4),
                    "total_ports": 6,
                    "price_per_kwh": 15.0 + (i % 5),
                    "avg_wait_estimate": float(i % 3 * 5.0),
                    "grid_load_kw": 50.0 + (i * 10),
                }
                for i in range(20)
            ]

        port_multiplier = 0.5 if self.scenario == "reduced_ports" else 1.0
        load_multiplier = 1.8 if self.scenario == "high_load" else 1.0
        traffic_multiplier = 1.6 if self.scenario == "high_traffic" else 1.0

        # Vectorized distance matrix calculation across all EVs x all Stations
        ev_coords = np.array(self.ev_positions if self.ev_positions else [(12.9716, 77.5946)], dtype=np.float64)
        if len(ev_coords) < self.num_evs:
            pad = np.tile([12.9716, 77.5946], (self.num_evs - len(ev_coords), 1))
            ev_coords = np.vstack([ev_coords, pad])
        ev_coords = ev_coords[:self.num_evs]

        st_coords = np.array([
            [float(s.get("lat") or s.get("latitude") or 12.9716), float(s.get("lon") or s.get("longitude") or 77.5946)]
            for s in all_stations
        ], dtype=np.float64)

        ev_lats = np.radians(ev_coords[:, 0])[:, None]
        ev_lons = np.radians(ev_coords[:, 1])[:, None]
        st_lats = np.radians(st_coords[:, 0])[None, :]
        st_lons = np.radians(st_coords[:, 1])[None, :]

        dlat = st_lats - ev_lats
        dlon = st_lons - ev_lons

        a = np.sin(dlat / 2.0)**2 + np.cos(ev_lats) * np.cos(st_lats) * np.sin(dlon / 2.0)**2
        dist_matrix = 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-12, 1.0 - a)))

        for ev_idx in range(self.num_evs):
            top_k_indices = np.argsort(dist_matrix[ev_idx])[:self.candidate_count]
            cands = []

            for idx in top_k_indices:
                s = all_stations[idx]
                sid = str(s.get("station_id") or f"st_{idx}")
                dist_km = float(dist_matrix[ev_idx, idx])

                total_ports = int(s.get("total_ports", 4))
                free_ports = max(0, int(int(s.get("available_ports", 2)) * port_multiplier))
                price = float(s.get("price_per_kwh", 16.0))
                avg_wait = float(s.get("avg_wait_estimate", 5.0)) * traffic_multiplier
                grid_load = float(s.get("grid_load_kw", 40.0)) * load_multiplier
                queue_len = max(0, total_ports - free_ports)
                travel_time = (dist_km / 25.0) * 60.0 * traffic_multiplier

                cands.append({
                    "station_id": sid,
                    "distance_km": float(dist_km),
                    "travel_time_min": float(travel_time),
                    "free_ports": free_ports,
                    "total_ports": total_ports,
                    "price_per_kwh": price,
                    "avg_wait_min": avg_wait,
                    "queue_len": queue_len,
                    "grid_load_kw": grid_load,
                })

            while len(cands) < self.candidate_count:
                cands.append({
                    "station_id": "dummy",
                    "distance_km": 999.0,
                    "travel_time_min": 999.0,
                    "free_ports": 0,
                    "total_ports": 1,
                    "price_per_kwh": 0.0,
                    "avg_wait_min": 999.0,
                    "queue_len": 99,
                    "grid_load_kw": 999.0,
                })

            self.current_candidates.append(cands)

    def get_single_agent_obs(self, ev_idx: int = 0) -> np.ndarray:
        tracked = self.vehicle_manager.list_tracked_vehicles()
        if ev_idx < len(tracked):
            vrec = tracked[ev_idx]
            lat, lon = self.ev_positions[ev_idx] if ev_idx < len(self.ev_positions) else (12.9716, 77.5946)
            ev_obs = [
                float(vrec.battery_pct),
                float(vrec.battery_capacity_kwh),
                float(vrec.remaining_range_km),
                float(lat),
                float(lon),
            ]
        else:
            ev_obs = [20.0, 40.0, 100.0, 12.9716, 77.5946]

        st_obs = []
        cands = self.current_candidates[ev_idx] if ev_idx < len(self.current_candidates) else []
        for c in cands:
            st_obs.extend([
                float(c["distance_km"]),
                float(c["travel_time_min"]),
                float(c["free_ports"]),
                float(c["total_ports"]),
                float(c["price_per_kwh"]),
                float(c["avg_wait_min"]),
            ])

        while len(st_obs) < self.candidate_count * self.station_feat_dim:
            st_obs.append(0.0)

        full_obs = np.array(ev_obs + st_obs, dtype=np.float32)
        return full_obs

    def get_multi_agent_obs(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """Returns local observations for each EV and global centralized observation."""
        local_obs_list = [self.get_single_agent_obs(i) for i in range(self.num_evs)]
        global_obs = np.concatenate(local_obs_list, axis=0)
        return local_obs_list, global_obs

    def calculate_reward_and_constraints(
        self, ev_idx: int, action_idx: int
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        action_idx = max(0, min(int(action_idx), self.candidate_count - 1))
        cands = self.current_candidates[ev_idx]
        chosen = cands[action_idx]

        tracked = self.vehicle_manager.list_tracked_vehicles()
        vrec = tracked[ev_idx] if ev_idx < len(tracked) else None
        battery_pct = float(vrec.battery_pct) if vrec else 20.0
        battery_cap = float(vrec.battery_capacity_kwh) if vrec else 40.0
        remaining_range = float(vrec.remaining_range_km) if vrec else 100.0

        dist = float(chosen["distance_km"])
        travel_time = float(chosen["travel_time_min"])
        free_ports = int(chosen["free_ports"])
        price = float(chosen["price_per_kwh"])
        avg_wait = float(chosen["avg_wait_min"])
        queue_len = int(chosen["queue_len"])
        grid_load = float(chosen["grid_load_kw"])

        # Rewards & Penalties
        reward = 0.0

        # Completion / Success reward
        is_success = dist < remaining_range
        is_completed = is_success and free_ports > 0

        if is_success:
            reward += 30.0
        if is_completed:
            reward += 20.0
        else:
            reward -= 20.0  # failed/invalid rec penalty

        reward += float(free_ports) * 0.5
        reward -= float(avg_wait) * 0.5
        reward -= float(dist) * 0.2
        
        energy_need = energy_needed_kwh(battery_pct, 100.0, battery_cap)
        reward -= float(energy_need * price) * 0.05
        reward -= float(queue_len) * 1.0
        reward -= float(grid_load) * 0.005

        # Unsafe SOC penalty
        if remaining_range < dist * 1.2:
            reward -= 50.0

        # Constraint evaluation for Constrained MAPPO
        constraints = {
            "c_unsafe_soc": 1.0 if remaining_range < dist * 1.2 else 0.0,
            "c_port_capacity": 1.0 if free_ports == 0 else 0.0,
            "c_queue_limit": 1.0 if avg_wait > 15.0 or queue_len > 3 else 0.0,
            "c_excess_detour": 1.0 if dist > 15.0 else 0.0,
            "c_grid_overload": 1.0 if grid_load > 120.0 else 0.0,
        }

        total_constraint_violations = sum(constraints.values())

        metrics = {
            "is_success": is_success,
            "is_completed": is_completed,
            "dist_km": dist,
            "travel_time_min": travel_time,
            "wait_time_sec": avg_wait * 60.0,
            "charging_cost": energy_need * price,
            "energy_consumed_kwh": dist * 0.20,
            "free_ports": free_ports,
            "queue_len": queue_len,
            "grid_load_kw": grid_load,
            "constraint_violations": total_constraint_violations,
            "station_id": chosen["station_id"],
        }

        return reward, constraints, metrics

    def step(self, actions: Union[int, List[int], np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1
        if isinstance(actions, (int, np.integer)):
            action_list = [int(actions)] * self.num_evs
        else:
            action_list = [int(a) for a in actions]
            while len(action_list) < self.num_evs:
                action_list.append(0)

        total_reward = 0.0
        all_metrics = []
        all_constraints = []

        for ev_idx in range(self.num_evs):
            r, c, m = self.calculate_reward_and_constraints(ev_idx, action_list[ev_idx])
            total_reward += r
            all_constraints.append(c)
            all_metrics.append(m)

        mean_reward = total_reward / max(1, self.num_evs)
        done = self.step_count >= self.max_steps

        # Re-sample candidates for next step
        self._sample_candidates_all_evs()
        obs = self.get_single_agent_obs(0)

        info = {
            "step_metrics": all_metrics,
            "constraints": all_constraints,
            "mean_reward": mean_reward,
        }

        return obs, float(mean_reward), done, False, info

    def close(self) -> None:
        if self._owns_station_manager and self.station_manager:
            try:
                self.station_manager.close()
            except Exception:
                pass
