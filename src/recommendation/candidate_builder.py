from __future__ import annotations

import math
from typing import Any

from src.recommendation.engine import StationCandidate


def build_candidates(vehicle_state: dict[str, Any], stations_lookup: dict[str, dict[str, Any]], net: Any | None = None, sumolib_net_graph: Any | None = None) -> list[StationCandidate]:
    """Create candidate station objects from a station lookup.

    This implementation uses the station lookup directly and keeps the interface open for a live
    SUMO-backed implementation in later phases.
    """
    candidates: list[StationCandidate] = []
    ev_lat = float(vehicle_state.get("lat", 12.9716) or 12.9716)
    ev_lon = float(vehicle_state.get("lon", 77.5946) or 77.5946)

    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))
        return 6371.0 * c

    for station_id, data in stations_lookup.items():
        lat = data.get("lat")
        lon = data.get("lon")
        if lat is not None and lon is not None:
            distance_km = float(_haversine_km(ev_lat, ev_lon, float(lat), float(lon)))
        else:
            distance_km = float(data.get("distance_km", 1e6))
        avg_speed_kmph = float(data.get("avg_speed_kmph", 24.0) or 24.0)
        travel_time_min = float(data.get("travel_time_min") or ((distance_km / max(1.0, avg_speed_kmph)) * 60.0))
        free_ports = int(data.get("free_ports", 1))
        total_ports = max(int(data.get("total_ports", 1)), 1)
        power_kw = float(data.get("power_kw", 22.0))
        if distance_km <= 0.0 or distance_km >= 1e6:
            continue
        candidates.append(
            StationCandidate(
                station_id=station_id,
                distance_km=distance_km,
                travel_time_min=travel_time_min,
                free_ports=free_ports,
                total_ports=total_ports,
                power_kw=power_kw,
            )
        )
    return candidates
