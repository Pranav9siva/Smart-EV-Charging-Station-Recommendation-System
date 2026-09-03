from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ChargingStation:
    station_id: str
    capacity: int
    occupancy: int
    waiting_time: float

    @property
    def available_spots(self) -> int:
        return max(0, self.capacity - self.occupancy)

    def is_available(self) -> bool:
        return self.available_spots > 0

    def reserve_slot(self) -> bool:
        if self.is_available():
            self.occupancy += 1
            return True
        return False

    def release_slot(self) -> None:
        self.occupancy = max(0, self.occupancy - 1)


class StationManager:
    def __init__(self, stations: list[ChargingStation]) -> None:
        self.stations = {station.station_id: station for station in stations}

    def list_all_stations(self) -> list[dict[str, Any]]:
        return [
            {
                "station_id": station.station_id,
                "name": station.station_id,
                "lat": 0.0,
                "lon": 0.0,
                "operator": "legacy",
                "address": "",
                "grid_zone_id": "",
                "data_source": "legacy",
                "waiting_time": float(station.waiting_time),
            }
            for station in self.stations.values()
        ]

    def get_station(self, station_id: str) -> ChargingStation | None:
        return self.stations.get(station_id)

    def get_station_state(self, station_id: str) -> dict[str, Any] | None:
        station = self.get_station(station_id)
        if station is None:
            return None
        return {
            "station_id": station.station_id,
            "ports": [{"port_id": f"{station.station_id}-port", "status": "Available", "power_kw": 22.0}],
            "price_per_kwh": 0.3,
            "avg_wait_estimate": float(station.waiting_time),
        }

    def get_station_metrics(self, station_id: str) -> dict[str, Any] | None:
        station = self.get_station(station_id)
        if station is None:
            return None
        return {
            "station_id": station.station_id,
            "available_ports": station.available_spots,
            "total_ports": station.capacity,
            "avg_wait_estimate": float(station.waiting_time),
            "price_per_kwh": 0.3,
            "travel_time_min": float(station.waiting_time),
            "distance_km": 1.0,
        }

    def get_available_stations(self) -> list[ChargingStation]:
        return [station for station in self.stations.values() if station.is_available()]

    def choose_best_station(self) -> ChargingStation | None:
        available = self.get_available_stations()
        if not available:
            return None
        return min(available, key=lambda s: s.waiting_time)
