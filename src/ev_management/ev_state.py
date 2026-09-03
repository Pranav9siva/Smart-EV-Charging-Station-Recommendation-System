from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ElectricVehicle:
    vehicle_id: str
    battery_level: float
    destination: str
    preferred_station: str | None = None
    status: str = "waiting"


class EVManager:
    def __init__(self) -> None:
        self.vehicles: dict[str, ElectricVehicle] = {}

    def register_vehicle(self, vehicle: ElectricVehicle) -> None:
        self.vehicles[vehicle.vehicle_id] = vehicle

    def get_vehicle(self, vehicle_id: str) -> ElectricVehicle | None:
        return self.vehicles.get(vehicle_id)

    def update_vehicle_state(self, vehicle_id: str, **updates: object) -> None:
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None:
            return
        for key, value in updates.items():
            if hasattr(vehicle, key):
                setattr(vehicle, key, value)

    def list_vehicles(self) -> list[ElectricVehicle]:
        return list(self.vehicles.values())
