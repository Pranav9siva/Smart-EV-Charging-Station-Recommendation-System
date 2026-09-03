from __future__ import annotations

import math
from typing import Any, List, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.core import Env

from src.monitoring.metrics import update_metric
from src.station_management.manager import ChargingStationManager
from src.ev_management.vehicle_manager import VehicleManager, VehicleRecord
from src.ev_model.battery import energy_needed_kwh, remaining_range_km


class GymEVChargingEnv(Env):
    """Gymnasium environment for EV charging station selection.

    Observation:
      - vehicles: array of tracked vehicles (vcount x features)
      - stations: array of candidate stations (k x features)

    Action:
      - Discrete index selecting one candidate station

    Reward components (summed, not heuristic-driven):
      - negative waiting time (minimize)
      - negative distance (minimize)
      - negative charging cost (minimize)
      - positive available ports (maximize)
      - negative queue length (minimize)
      - negative grid_load (minimize)
      - battery safety penalty if remaining range insufficient
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        station_manager: Optional[ChargingStationManager] = None,
        tracked_vehicle_count: int = 10,
        candidate_count: int = 8,
    ) -> None:
        self._owns_station_manager = station_manager is None and bool(db_path)
        self.station_manager = station_manager or (ChargingStationManager(db_path) if db_path else None)
        self.tracked_vehicle_count = int(tracked_vehicle_count)
        self.candidate_count = int(candidate_count)

        # vehicles features: [battery_pct, battery_kwh, consumption_wh_per_km, remaining_range_km, lat, lon]
        self.vehicle_feat_dim = 6
        # station features: [distance_km, travel_time_min, free_ports, total_ports, price_per_kwh, avg_wait_min]
        self.station_feat_dim = 6

        self.episode_length = 0
        self.episode_reward = 0.0
        self.observation_space = spaces.Dict(
            {
                "vehicles": spaces.Box(low=-1e6, high=1e6, shape=(self.tracked_vehicle_count, self.vehicle_feat_dim), dtype=np.float32),
                "stations": spaces.Box(low=-1e6, high=1e6, shape=(self.candidate_count, self.station_feat_dim), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(self.candidate_count)

        # internal state
        self.vehicle_manager = VehicleManager()
        self.current_candidates: List[dict[str, Any]] = []
        self._query_lat: Optional[float] = None
        self._query_lon: Optional[float] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # re-generate tracked vehicles if none exist
        if len(self.vehicle_manager.list_vehicles()) < self.tracked_vehicle_count:
            self.vehicle_manager.generate_fleet(count=100, detail_count=self.tracked_vehicle_count)

        self.episode_length = 0
        self.episode_reward = 0.0
        self._sample_candidates()
        obs = self._build_observation()
        update_metric("episode_number", self.episode_length)
        return obs, {}

    def set_query_location(self, lat: float | None, lon: float | None) -> None:
        self._query_lat = float(lat) if lat is not None else None
        self._query_lon = float(lon) if lon is not None else None

    def _resolve_query_location(self) -> tuple[float, float]:
        if self._query_lat is not None and self._query_lon is not None:
            return float(self._query_lat), float(self._query_lon)
        vehicles = self.vehicle_manager.list_tracked_vehicles()
        if vehicles:
            # Fleet records do not currently persist geocoordinates; use Bengaluru center fallback.
            return 12.9716, 77.5946
        return 12.9716, 77.5946

    def _estimate_travel_time_min(self, distance_km: float, queue_len: int, avg_wait_min: float, grid_load_kw: float) -> float:
        base_speed_kmph = 24.0
        drive_min = (max(0.0, distance_km) / max(1.0, base_speed_kmph)) * 60.0
        congestion_penalty = min(20.0, max(0, queue_len) * 1.5)
        load_penalty = min(10.0, max(0.0, grid_load_kw) * 0.01)
        return max(0.1, drive_min + max(0.0, avg_wait_min) + congestion_penalty + load_penalty)

    def _sample_candidates(self) -> None:
        # Build candidate list from nearby station snapshots ordered by actual geodesic distance.
        self.current_candidates = []
        if not self.station_manager:
            # empty candidates
            self.current_candidates = [self._empty_candidate() for _ in range(self.candidate_count)]
            return

        qlat, qlon = self._resolve_query_location()
        stations = []
        nearby_method = getattr(self.station_manager, "get_nearby", None)
        if callable(nearby_method):
            for radius in (3.0, 6.0, 10.0, 20.0):
                stations = list(nearby_method(qlat, qlon, radius_km=radius))
                if len(stations) >= self.candidate_count:
                    break
        if not stations:
            stations = self.station_manager.list_all_stations()

        for s in stations:
            station_id = str(s.get("station_id") or "")
            if not station_id:
                continue
            metrics = self.station_manager.get_station_metrics(station_id) or {}
            distance_km = s.get("distance_km")
            if distance_km is None:
                lat = s.get("lat")
                lon = s.get("lon")
                if lat is not None and lon is not None:
                    # Fast geodesic approximation for candidate ranking.
                    dlat = math.radians(float(lat) - qlat)
                    dlon = math.radians(float(lon) - qlon)
                    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(qlat)) * math.cos(math.radians(float(lat))) * math.sin(dlon / 2) ** 2
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))
                    distance_km = 6371.0 * c
                else:
                    distance_km = metrics.get("distance_km", 1e6)

            free_ports = int(metrics.get("available_ports", 0) or 0)
            total_ports = int(metrics.get("total_ports", 1) or 1)
            queue_len = int(metrics.get("queue_length", max(0, total_ports - free_ports)) or 0)
            avg_wait_min = float(metrics.get("avg_wait_estimate", 0.0) or 0.0)
            grid_load_kw = float(metrics.get("power_kw", metrics.get("grid_load_kw", 0.0)) or 0.0)
            travel_time_min = metrics.get("travel_time_min")
            if travel_time_min is None:
                travel_time_min = self._estimate_travel_time_min(float(distance_km), queue_len, avg_wait_min, grid_load_kw)

            # Exclude invalid/unreachable candidates from normal recommendation path.
            if float(distance_km) <= 0.0 or float(distance_km) >= 1e6:
                continue
            if float(travel_time_min) <= 0.0 or float(travel_time_min) >= 1e6:
                continue

            cand = {
                "station_id": station_id,
                "distance_km": float(distance_km),
                "travel_time_min": float(travel_time_min),
                "free_ports": free_ports,
                "total_ports": max(1, total_ports),
                "price_per_kwh": float(metrics.get("price_per_kwh") or 0.0),
                "avg_wait_min": avg_wait_min,
                "queue_len": queue_len,
                "grid_load_kw": grid_load_kw,
                "charging_power_kw": grid_load_kw,
            }
            self.current_candidates.append(cand)
            if len(self.current_candidates) >= self.candidate_count:
                break

        self.current_candidates.sort(key=lambda item: (float(item.get("distance_km", 1e6)), float(item.get("travel_time_min", 1e6))))

        # pad
        while len(self.current_candidates) < self.candidate_count:
            self.current_candidates.append(self._empty_candidate())

    def _empty_candidate(self) -> dict[str, Any]:
        return {"station_id": "", "distance_km": 1e6, "travel_time_min": 1e6, "free_ports": 0, "total_ports": 0, "price_per_kwh": 0.0, "avg_wait_min": 1e6, "queue_len": 0, "grid_load_kw": 0.0}

    def _build_observation(self) -> dict:
        vehicles = self.vehicle_manager.list_tracked_vehicles()
        veh_arr = np.zeros((self.tracked_vehicle_count, self.vehicle_feat_dim), dtype=np.float32)
        for i in range(self.tracked_vehicle_count):
            if i < len(vehicles):
                v = vehicles[i]
                veh_arr[i, 0] = float(v.battery_pct)
                veh_arr[i, 1] = float(v.battery_capacity_kwh)
                # best-effort consumption value (unknown here) -> 200
                veh_arr[i, 2] = 200.0
                veh_arr[i, 3] = float(v.remaining_range_km)
                veh_arr[i, 4] = 0.0
                veh_arr[i, 5] = 0.0
            else:
                veh_arr[i] = np.zeros(self.vehicle_feat_dim, dtype=np.float32)

        st_arr = np.zeros((self.candidate_count, self.station_feat_dim), dtype=np.float32)
        for i in range(self.candidate_count):
            c = self.current_candidates[i]
            st_arr[i, 0] = float(c.get("distance_km", 0.0))
            st_arr[i, 1] = float(c.get("travel_time_min", 0.0))
            st_arr[i, 2] = float(c.get("free_ports", 0))
            st_arr[i, 3] = float(c.get("total_ports", 0))
            st_arr[i, 4] = float(c.get("price_per_kwh", 0.0))
            st_arr[i, 5] = float(c.get("avg_wait_min", 0.0))

        return {"vehicles": veh_arr, "stations": st_arr}

    def step(self, action):
        # action is an index into current_candidates
        chosen_idx = int(action) if action is not None else 0
        chosen_idx = max(0, min(chosen_idx, self.candidate_count - 1))
        chosen = self.current_candidates[chosen_idx]

        # compute reward across tracked vehicles (aggregate)
        reward = 0.0
        vehicles = self.vehicle_manager.list_tracked_vehicles()
        for v in vehicles:
            # safety: if remaining range insufficient for distance -> heavy penalty
            dist = chosen.get("distance_km", 1e6)
            if v.remaining_range_km < dist * 1.2:
                reward -= 50.0

            # waiting time: shorter is better
            reward -= float(chosen.get("avg_wait_min", 0.0)) * 0.5

            # distance: penalize by km
            reward -= float(dist) * 0.2

            # cost: energy need to full * price
            energy_need = energy_needed_kwh(v.battery_pct, 100.0, v.battery_capacity_kwh)
            reward -= energy_need * float(chosen.get("price_per_kwh", 0.0)) * 0.1

            # available ports: encourage more
            reward += float(chosen.get("free_ports", 0)) * 0.5

            # queue length: penalize
            reward -= float(chosen.get("queue_len", 0)) * 1.0

            # grid load: penalize high load
            reward -= float(chosen.get("grid_load_kw", 0.0)) * 0.01

        self.episode_length += 1
        self.episode_reward += float(reward)
        update_metric("ppo_reward", float(reward))
        update_metric("episode_reward", self.episode_reward)
        update_metric("episode_length", self.episode_length)
        update_metric("ppo_reward_avg", self.episode_reward / max(1, self.episode_length))

        self._sample_candidates()
        obs = self._build_observation()
        done = False
        info = {"chosen_station": chosen.get("station_id", "")}
        return obs, float(reward), done, False, info

    def render(self):
        print("GymEVChargingEnv: tracked vehicles:")
        for v in self.vehicle_manager.list_tracked_vehicles():
            print(v.to_dict())

    def close(self) -> None:
        if self.station_manager and hasattr(self.station_manager, "close"):
            # Only close the station manager if this environment owns it; otherwise
            # preserve the shared connection for the caller that injected it.
            if getattr(self, "_owns_station_manager", False):
                try:
                    self.station_manager.close()
                except Exception:
                    pass
