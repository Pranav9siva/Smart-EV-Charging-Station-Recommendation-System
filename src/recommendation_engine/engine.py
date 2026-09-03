from __future__ import annotations

import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None

try:
    from src.route_planning.network_graph import NetworkGraph
except Exception:
    NetworkGraph = None
try:
    from src.rl_env.gym_ev_charging_env import GymEVChargingEnv
except Exception:
    GymEVChargingEnv = None
from src.monitoring.metrics import update_metric
from src.station_management.station_state import ChargingStation, StationManager
from src.ev_model.battery import energy_needed_kwh


class RecommendationResult(dict):
    def __getattr__(self, name: str):
        if name in self:
            return self[name]
        if name == "station_id":
            return self.get("station")
        raise AttributeError(name)


class RecommendationEngine:
    """Recommendation engine using a deterministic weighted multi-factor objective.

    Station diversity is enforced via a per-station assignment counter so that
    multiple EVs are not all sent to the same station simultaneously.
    """

    def __init__(self, station_manager: StationManager, network: NetworkGraph | None = None, model_path: str | None = None) -> None:
        self.station_manager = station_manager
        self.network = network
        self.model_path = model_path
        self._ppo_model = None
        self.model_load_count = 0
        self.weights = {
            "distance": 0.25,
            "travel_time": 0.25,
            "cost": 0.15,
            "wait": 0.15,
            "queue": 0.10,
            "ports": 0.07,
            "power": 0.03,
        }
        # tracks how many EVs are currently routed to each station; enforces diversity
        self._station_assignment_count: dict[str, int] = {}
        self.max_evs_per_station: int = 3
        if model_path:
            self._ensure_model_loaded(model_path)

    def configure_weights(self, weights: Dict[str, float]) -> None:
        for key, value in (weights or {}).items():
            if key in self.weights:
                self.weights[key] = float(value)

    def _resolve_model_path(self, model_path: str | None) -> str | None:
        if not model_path:
            return self.model_path
        candidate_path = Path(str(model_path))
        if not candidate_path.is_absolute():
            root = Path(__file__).resolve().parents[2]
            candidate_path = (root / candidate_path).resolve()
            if not candidate_path.exists():
                candidate_path = (root / "models" / candidate_path.name).resolve()
        return str(candidate_path)

    def _ensure_model_loaded(self, model_path: str | None = None):
        if PPO is None:
            return None
        resolved_path = self._resolve_model_path(model_path)
        if not resolved_path:
            return None
        if self._ppo_model is not None and self.model_path == resolved_path:
            return self._ppo_model

        try:
            self._ppo_model = PPO.load(resolved_path)
        except Exception:
            return None

        self.model_path = resolved_path
        self.model_load_count += 1
        return self._ppo_model

    def _recommend_heuristic(self, vehicle_state: Dict[str, Any] | None = None, origin: str | None = None, destination: str | None = None) -> Dict[str, Any]:
        candidates: list[Dict[str, Any]] = []
        if hasattr(self.station_manager, "list_all_stations"):
            try:
                candidates = list(self.station_manager.list_all_stations())
            except Exception:
                candidates = []
        elif hasattr(self.station_manager, "get_available_stations"):
            try:
                available = self.station_manager.get_available_stations()
                candidates = [
                    {
                        "station_id": station.station_id,
                        "waiting_time": float(getattr(station, "waiting_time", 0.0)),
                        "distance_km": 1.0,
                        "total_ports": int(getattr(station, "capacity", 1)),
                        "available_ports": int(getattr(station, "available_spots", 0)),
                    }
                    for station in available
                ]
            except Exception:
                candidates = []
        elif hasattr(self.station_manager, "stations"):
            for station in getattr(self.station_manager, "stations", {}).values():
                candidates.append(
                    {
                        "station_id": station.station_id,
                        "waiting_time": float(getattr(station, "waiting_time", 0.0)),
                        "distance_km": 1.0,
                        "total_ports": int(getattr(station, "capacity", 1)),
                        "available_ports": int(getattr(station, "available_spots", 0)),
                    }
                )

        if not candidates:
            return RecommendationResult({
                "station": "",
                "station_name": "",
                "selected_station": "",
                "travel_distance": 0.0,
                "travel_time": 0.0,
                "waiting_time": 0.0,
                "queue_length": 0,
                "charging_cost": 0.0,
                "available_ports": 0,
                "charging_speed_kw": 0.0,
                "grid_load": 0.0,
                "recommendation_score": 0.0,
                "recommendation_reason": "heuristic:fallback",
                "ppo_reward": None,
            })

        chosen = min(
            candidates,
            key=lambda item: (
                float(item.get("waiting_time", 0.0)),
                float(item.get("distance_km", 0.0)),
                -int(item.get("available_ports", 0)),
            ),
        )

        station_name = chosen.get("name") or chosen.get("station_name") or chosen.get("station_id", "")
        return RecommendationResult({
            "station": chosen.get("station_id", ""),
            "station_name": station_name,
            "selected_station": chosen.get("station_id", ""),
            "travel_distance": float(chosen.get("distance_km", 0.0)),
            "travel_time": float(chosen.get("waiting_time", 0.0)),
            "waiting_time": float(chosen.get("waiting_time", 0.0)),
            "queue_length": max(0, int(chosen.get("total_ports", 1)) - int(chosen.get("available_ports", 0))),
            "charging_cost": 0.0,
            "available_ports": int(chosen.get("available_ports", 0)),
            "charging_speed_kw": 22.0,
            "grid_load": 0.0,
            "recommendation_score": 0.0,
            "recommendation_reason": f"heuristic:origin={origin or 'unknown'} dest={destination or 'unknown'}",
            "ppo_reward": None,
        })

    def _resolve_station_id(self, station_id: str | None, *, strict: bool = False) -> str | None:
        if not station_id:
            return None
        raw_id = str(station_id).strip()
        if not raw_id:
            return None
        resolve_method = getattr(self.station_manager, "resolve_station_id", None)
        if callable(resolve_method):
            resolved = resolve_method(raw_id)
            if resolved:
                return str(resolved)

        list_method = getattr(self.station_manager, "list_all_stations", None)
        if callable(list_method):
            stations = list(list_method())
            for station in stations:
                station_id_value = str(station.get("station_id") or "")
                if station_id_value and station_id_value == raw_id:
                    return station_id_value
                if station.get("name") and str(station.get("name")).lower() == raw_id.lower():
                    return station_id_value
            if not strict and stations:
                first_station = str(stations[0].get("station_id") or "")
                if first_station:
                    return first_station

        metrics_method = getattr(self.station_manager, "get_station_metrics", None)
        if callable(metrics_method):
            metrics = metrics_method(raw_id)
            if metrics:
                return str(metrics.get("station_id") or raw_id)

        return None if strict else raw_id

    def _get_station_record(self, station_id: str | None) -> tuple[str | None, dict[str, Any] | None]:
        resolved_id = self._resolve_station_id(station_id, strict=False)
        if resolved_id:
            metrics_method = getattr(self.station_manager, "get_station_metrics", None)
            if callable(metrics_method):
                metrics = metrics_method(resolved_id)
                if metrics:
                    return str(metrics.get("station_id") or resolved_id), metrics
            state_method = getattr(self.station_manager, "get_station_state", None)
            if callable(state_method):
                state = state_method(resolved_id)
                if state:
                    return str(state.get("station_id") or resolved_id), state
            list_method = getattr(self.station_manager, "list_all_stations", None)
            if callable(list_method):
                for station in list_method():
                    if str(station.get("station_id") or "") == resolved_id:
                        return str(station.get("station_id") or resolved_id), station

        list_method = getattr(self.station_manager, "list_all_stations", None)
        if callable(list_method):
            for station in list_method():
                station_id_value = str(station.get("station_id") or "")
                if station_id_value:
                    return station_id_value, station

        return None, None

    def _score_recommendation(self, chosen: dict[str, Any], ppo_reward: float | None) -> float:
        if ppo_reward is not None:
            return float(ppo_reward)
        distance = float(chosen.get("distance_km", 0.0) or 0.0)
        travel_time = float(chosen.get("travel_time_min", 0.0) or 0.0)
        waiting = float(chosen.get("avg_wait_min", 0.0) or 0.0)
        price = float(chosen.get("price_per_kwh", 0.0) or 0.0)
        queue = float(chosen.get("queue_len", 0) or 0.0)
        free_ports = float(chosen.get("free_ports", 0) or 0.0)
        return -(distance * 0.2 + travel_time * 0.3 + waiting * 0.2 + price * 5.0 + queue * 0.5 - free_ports * 0.4)

    def _normalize(self, values: list[float], value: float) -> float:
        if not values:
            return 0.0
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            return 0.0
        return (value - lo) / (hi - lo)

    def _compute_weighted_scores(self, vehicle_state: Dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []

        battery_pct = float(vehicle_state.get("battery_pct", 0.0) or 0.0)
        battery_kwh = float(vehicle_state.get("battery_capacity_kwh", vehicle_state.get("battery_kwh", 0.0)) or 0.0)
        energy_need = energy_needed_kwh(battery_pct, 100.0, battery_kwh)

        distances = [float(item.get("distance_km", 0.0) or 0.0) for item in candidates]
        travel_times = [float(item.get("travel_time_min", 0.0) or 0.0) for item in candidates]
        waits = [float(item.get("avg_wait_min", 0.0) or 0.0) for item in candidates]
        prices = [float(item.get("price_per_kwh", 0.0) or 0.0) for item in candidates]
        queues = [float(item.get("queue_len", 0.0) or 0.0) for item in candidates]
        free_ports = [float(item.get("free_ports", 0.0) or 0.0) for item in candidates]
        powers = [float(item.get("charging_power_kw", item.get("grid_load_kw", 0.0)) or 0.0) for item in candidates]

        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            distance = float(candidate.get("distance_km", 0.0) or 0.0)
            travel_time = float(candidate.get("travel_time_min", 0.0) or 0.0)
            wait_min = float(candidate.get("avg_wait_min", 0.0) or 0.0)
            price = float(candidate.get("price_per_kwh", 0.0) or 0.0)
            queue_len = float(candidate.get("queue_len", 0.0) or 0.0)
            ports = float(candidate.get("free_ports", 0.0) or 0.0)
            power = float(candidate.get("charging_power_kw", candidate.get("grid_load_kw", 0.0)) or 0.0)

            norm_distance = self._normalize(distances, distance)
            norm_travel_time = self._normalize(travel_times, travel_time)
            norm_cost = self._normalize([p * max(0.0, energy_need) for p in prices], price * max(0.0, energy_need))
            norm_wait = self._normalize(waits, wait_min)
            norm_queue = self._normalize(queues, queue_len)
            norm_ports = self._normalize(free_ports, ports)
            norm_power = self._normalize(powers, power)

            weighted_score = (
                self.weights["distance"] * norm_distance
                + self.weights["travel_time"] * norm_travel_time
                + self.weights["cost"] * norm_cost
                + self.weights["wait"] * norm_wait
                + self.weights["queue"] * norm_queue
                - self.weights["ports"] * norm_ports
                - self.weights["power"] * norm_power
            )

            scored.append(
                {
                    **candidate,
                    "weighted_score": float(weighted_score),
                    "normalized": {
                        "distance": norm_distance,
                        "travel_time": norm_travel_time,
                        "cost": norm_cost,
                        "wait": norm_wait,
                        "queue": norm_queue,
                        "ports": norm_ports,
                        "power": norm_power,
                    },
                }
            )

        scored.sort(key=lambda item: (float(item.get("weighted_score", 1e9)), float(item.get("distance_km", 1e9))))
        return scored

    def _build_recommendation_reason(self, chosen: dict[str, Any], vehicle_state: Dict[str, Any]) -> str:
        normalized = chosen.get("normalized") or {}
        factors: list[str] = []
        if float(normalized.get("wait", 1.0) or 1.0) <= 0.35 or float(chosen.get("avg_wait_min", 0.0) or 0.0) <= 1.0:
            factors.append("low wait")
        if int(chosen.get("free_ports", 0) or 0) > 0:
            factors.append("available port")
        if float(normalized.get("distance", 1.0) or 1.0) <= 0.35:
            factors.append("shorter distance")
        if float(normalized.get("cost", 1.0) or 1.0) <= 0.35:
            factors.append("lower price")
        if not factors:
            factors.append("best weighted combination")

        battery_pct = vehicle_state.get("battery_pct")
        battery_text = f"; SOC {float(battery_pct):.1f}%" if battery_pct is not None else ""
        return (
            f"Best score: {' + '.join(factors)} "
            f"(distance {float(chosen.get('distance_km', 0.0) or 0.0):.1f} km, "
            f"price {float(chosen.get('price_per_kwh', 0.0) or 0.0):.2f}/kWh, "
            f"wait {float(chosen.get('avg_wait_min', 0.0) or 0.0):.1f} min, "
            f"{int(chosen.get('free_ports', 0) or 0)} free ports{battery_text})."
        )

    def recommend(self, vehicle_state: Dict[str, Any] | str, model_path: str | None = None, candidate_count: int = 8, tracked_vehicle_count: int = 10) -> Dict[str, Any]:
        """Recommend a charging station from live station metrics using a deterministic weighted objective."""
        if isinstance(vehicle_state, str):
            return self._recommend_heuristic(origin=vehicle_state, destination=str(model_path or ""))

        model = self._ensure_model_loaded(model_path)
        if model is not None and GymEVChargingEnv is not None and isinstance(vehicle_state, dict):
            try:
                env = GymEVChargingEnv(
                    station_manager=self.station_manager,
                    tracked_vehicle_count=tracked_vehicle_count,
                    candidate_count=candidate_count,
                )
                if "lat" in vehicle_state and "lon" in vehicle_state:
                    env.set_query_location(vehicle_state["lat"], vehicle_state["lon"])
                obs, _ = env.reset()
                action_out, _ = model.predict(obs, deterministic=True)
                if isinstance(action_out, (list, tuple, np.ndarray)):
                    action = int(action_out[0])
                else:
                    action = int(action_out)
                if 0 <= action < len(env.current_candidates):
                    cand = env.current_candidates[action]
                    sid = cand.get("station_id")
                    if sid:
                        sel_id, st_rec = self._get_station_record(sid)
                        if not sel_id:
                            sel_id = sid
                        st_name = (st_rec.get("name") if isinstance(st_rec, dict) else None) or (st_rec.get("station_name") if isinstance(st_rec, dict) else None) or sel_id
                        return RecommendationResult({
                            "station": sel_id,
                            "station_name": st_name,
                            "selected_station": sel_id,
                            "travel_distance": float(cand.get("distance_km", 0.0)),
                            "travel_time": float(cand.get("travel_time_min", 0.0)),
                            "waiting_time": float(cand.get("avg_wait_min", 0.0)),
                            "queue_length": int(cand.get("queue_len", 0)),
                            "charging_cost": float(cand.get("price_per_kwh", 0.0)) * 20.0,
                            "available_ports": int(cand.get("free_ports", 0)),
                            "charging_speed_kw": float(cand.get("charging_power_kw", 22.0)),
                            "grid_load": float(cand.get("grid_load_kw", 0.0)),
                            "recommendation_score": float(self._score_recommendation(cand, 10.0)),
                            "recommendation_reason": "PPO Model Policy Selection",
                            "ppo_reward": 10.0,
                        })
            except Exception:
                pass

        if not hasattr(self.station_manager, "list_all_stations") and not hasattr(self.station_manager, "get_station_metrics"):
            return self._recommend_heuristic(vehicle_state=vehicle_state)

        lat = float(vehicle_state.get("lat", 12.9716) or 12.9716)
        lon = float(vehicle_state.get("lon", 77.5946) or 77.5946)

        stations: list[dict[str, Any]] = []
        if hasattr(self.station_manager, "list_all_stations"):
            try:
                stations = list(self.station_manager.list_all_stations())
            except Exception:
                stations = []

        candidates: list[dict[str, Any]] = []
        for station in stations:
            station_id = str(station.get("station_id") or "")
            if not station_id:
                continue
            # exclude synthetic simulation scaffolding stations from real recommendations
            sid_lower = station_id.lower()
            if (sid_lower.startswith("sim_") or sid_lower.startswith("cs_net_")
                    or sid_lower.startswith("phase1c_") or sid_lower.startswith("sim_station_")):
                continue
            station_metrics = None
            if hasattr(self.station_manager, "get_station_metrics"):
                try:
                    station_metrics = self.station_manager.get_station_metrics(station_id) or {}
                except Exception:
                    station_metrics = {}
            if not station_metrics and isinstance(station, dict):
                station_metrics = station

            station_lat = station.get("lat")
            station_lon = station.get("lon")
            if station_lat is None or station_lon is None:
                station_lat = station_metrics.get("lat")
                station_lon = station_metrics.get("lon")

            dist_km = float(station.get("distance_km") or station_metrics.get("distance_km") or 0.0)
            if dist_km <= 0.0 and station_lat is not None and station_lon is not None:
                dlat = math.radians(float(station_lat) - lat)
                dlon = math.radians(float(station_lon) - lon)
                a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(float(station_lat))) * math.sin(dlon / 2) ** 2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))
                dist_km = 6371.0 * c

            if dist_km <= 0.0:
                continue

            if dist_km <= 0.0 or dist_km >= 1e6:
                continue

            travel_time_min = float(station_metrics.get("travel_time_min") or station.get("travel_time_min") or 0.0)
            if travel_time_min <= 0.0:
                travel_time_min = max(0.1, (dist_km / 24.0) * 60.0)

            free_ports = int(station_metrics.get("available_ports") or station.get("free_ports") or 0)
            total_ports = max(1, int(station_metrics.get("total_ports") or station.get("total_ports") or 1))
            queue_len = int(station_metrics.get("queue_length") or station.get("queue_len") or max(0, total_ports - free_ports))
            avg_wait_min = float(station_metrics.get("avg_wait_estimate") or station.get("avg_wait_min") or 0.0)
            price_per_kwh = float(station_metrics.get("price_per_kwh") or station.get("price_per_kwh") or 0.0)
            grid_load_kw = float(station_metrics.get("power_kw") or station_metrics.get("grid_load_kw") or station.get("grid_load_kw") or 0.0)
            charging_power_kw = float(station_metrics.get("charging_power_kw") or grid_load_kw or 22.0)

            candidates.append(
                {
                    "station_id": station_id,
                    "distance_km": float(dist_km),
                    "travel_time_min": float(travel_time_min),
                    "free_ports": free_ports,
                    "total_ports": total_ports,
                    "price_per_kwh": price_per_kwh,
                    "avg_wait_min": avg_wait_min,
                    "queue_len": queue_len,
                    "grid_load_kw": grid_load_kw,
                    "charging_power_kw": charging_power_kw,
                }
            )

        if not candidates:
            return self._recommend_heuristic(vehicle_state=vehicle_state)

        scored_candidates = self._compute_weighted_scores(vehicle_state, candidates)

        # Station diversity: skip stations already at capacity unless no alternatives exist
        chosen = None
        for c in scored_candidates:
            sid = c.get("station_id", "")
            if self._station_assignment_count.get(sid, 0) < self.max_evs_per_station:
                chosen = c
                break
        if chosen is None:
            chosen = scored_candidates[0] if scored_candidates else candidates[0]

        selected_station_id, station_record = self._get_station_record(chosen.get("station_id"))
        if not selected_station_id and isinstance(chosen, dict):
            selected_station_id = str(chosen.get("station_id") or "")
        # track assignment for diversity enforcement
        if selected_station_id:
            self._station_assignment_count[selected_station_id] = (
                self._station_assignment_count.get(selected_station_id, 0) + 1
            )

        station_metrics = None
        if selected_station_id:
            station_metrics = self.station_manager.get_station_metrics(selected_station_id) if hasattr(self.station_manager, "get_station_metrics") else None
        if station_metrics is None and isinstance(station_record, dict):
            station_metrics = station_record
        station_name = None
        if isinstance(station_metrics, dict):
            station_name = station_metrics.get("name") or station_metrics.get("station_name") or selected_station_id
        if not station_name and selected_station_id:
            station_name = selected_station_id

        battery_pct = float(vehicle_state.get("battery_pct", 0.0) or 0.0)
        battery_kwh = float(vehicle_state.get("battery_capacity_kwh", vehicle_state.get("battery_kwh", 0.0)) or 0.0)
        energy_need = energy_needed_kwh(battery_pct, 100.0, battery_kwh)
        price = float(chosen.get("price_per_kwh", 0.0))
        charging_cost = energy_need * price

        charging_speed_kw = 0.0
        st_state = self.station_manager.get_station_state(selected_station_id) if selected_station_id else None
        if st_state:
            ports = st_state.get("ports", [])
            if ports:
                charging_speed_kw = max(float(p.get("power_kw", 22.0)) for p in ports)

        update_metric('policy_inference_latency', 0.0)
        update_metric('decision_time', 0.0)

        recommendation_score = self._score_recommendation(chosen, None)
        return RecommendationResult({
            "station": selected_station_id or "",
            "station_name": station_name,
            "selected_station": selected_station_id or "",
            "travel_distance": float(chosen.get("distance_km", 0.0)),
            "travel_time": float(chosen.get("travel_time_min", 0.0)),
            "waiting_time": float(chosen.get("avg_wait_min", 0.0)),
            "queue_length": int(chosen.get("queue_len", 0)),
            "charging_cost": float(charging_cost),
            "available_ports": int(chosen.get("free_ports", 0)),
            "charging_speed_kw": float(charging_speed_kw),
            "grid_load": float(chosen.get("grid_load_kw", 0.0)),
            "recommendation_score": float(chosen.get("weighted_score", recommendation_score)),
            "recommendation_reason": self._build_recommendation_reason(chosen, vehicle_state),
            "ppo_reward": None,
            "score_breakdown": chosen.get("normalized", {}),
            "weights": dict(self.weights),
        })

