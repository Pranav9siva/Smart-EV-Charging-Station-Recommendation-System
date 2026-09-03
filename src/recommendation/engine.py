from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.ev_model.battery import charge_time_minutes, energy_needed_kwh, remaining_range_km


@dataclass
class StationCandidate:
    station_id: str
    distance_km: float
    travel_time_min: float
    free_ports: int
    total_ports: int
    power_kw: float


@dataclass
class Recommendation:
    station_id: str
    score: float
    reachable: bool
    travel_time_min: float
    queue_wait_min: float
    charge_time_min: float
    total_time_min: float
    breakdown: dict[str, float] = field(default_factory=dict)


class RecommendationEngine:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = {
            "time": 0.45,
            "distance": 0.15,
            "availability": 0.25,
            "battery_safety": 0.15,
        }
        if weights:
            self.weights.update(weights)

    def score_station(self, vehicle_state: dict[str, Any], candidate: StationCandidate, target_soc_pct: float = 80.0) -> Recommendation:
        soc_pct = float(vehicle_state.get("soc_pct", 0.0))
        battery_kwh = float(vehicle_state.get("battery_kwh", 0.0))
        consumption_wh_per_km = float(vehicle_state.get("consumption_wh_per_km", 0.0))

        reachable = remaining_range_km(soc_pct, battery_kwh, consumption_wh_per_km, reserve_pct=10.0) >= candidate.distance_km * 1.15
        if not reachable:
            unreachable_penalty = 5.0
        else:
            unreachable_penalty = 0.0

        availability_ratio = 1.0 - (candidate.free_ports / max(1, candidate.total_ports))
        queue_wait_min = 0.0 if candidate.free_ports > 0 else max(1.0, (candidate.total_ports - candidate.free_ports) * 5.0 / max(1, candidate.total_ports))

        energy_needed = energy_needed_kwh(soc_pct, target_soc_pct, battery_kwh)
        charge_time_min = charge_time_minutes(energy_needed, candidate.power_kw)
        total_time_min = candidate.travel_time_min + queue_wait_min + charge_time_min

        time_norm = total_time_min / max(1.0, 60.0)
        distance_norm = candidate.distance_km / max(1.0, 20.0)
        availability_norm = availability_ratio
        battery_norm = 0.0 if reachable else 1.0

        if self.weights.get("time", 0.0) == 0.0 and candidate.travel_time_min > 0:
            time_norm = candidate.travel_time_min / max(1.0, 30.0)

        score = (
            self.weights["time"] * time_norm
            + self.weights["distance"] * distance_norm
            + self.weights["availability"] * availability_norm
            + self.weights["battery_safety"] * (battery_norm + unreachable_penalty / 5.0)
        )

        breakdown = {
            "time": time_norm,
            "distance": distance_norm,
            "availability": availability_norm,
            "battery_safety": battery_norm + unreachable_penalty / 5.0,
        }

        return Recommendation(
            station_id=candidate.station_id,
            score=score,
            reachable=reachable,
            travel_time_min=candidate.travel_time_min,
            queue_wait_min=queue_wait_min,
            charge_time_min=charge_time_min,
            total_time_min=total_time_min,
            breakdown=breakdown,
        )

    def recommend(self, vehicle_state: dict[str, Any], candidates: list[StationCandidate], top_k: int = 3) -> list[Recommendation]:
        scored = [self.score_station(vehicle_state, candidate) for candidate in candidates]
        scored.sort(key=lambda item: (item.score, item.reachable is False, item.total_time_min))
        reachable = [item for item in scored if item.reachable]
        if reachable:
            top = reachable[:top_k]
            if len(top) < top_k:
                top.extend(item for item in scored if item not in top and item.reachable is False)  # type: ignore[arg-type]
            return top[:top_k]
        return scored[:top_k]
