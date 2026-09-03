from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ev_model.battery import energy_needed_kwh, remaining_range_km
from src.monitoring.metrics import update_metric


@dataclass
class VehicleRecord:
    vehicle_id: str
    source: str
    destination: str
    battery_capacity_kwh: float
    battery_pct: float
    remaining_range_km: float
    charging_requirement_kwh: float
    tracked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "source": self.source,
            "destination": self.destination,
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "battery_pct": self.battery_pct,
            "remaining_range_km": self.remaining_range_km,
            "charging_requirement_kwh": self.charging_requirement_kwh,
            "tracked": self.tracked,
        }


class VehicleManager:
    def __init__(self, vehicles: list[VehicleRecord] | None = None, detail_count: int = 10) -> None:
        self.vehicles: dict[str, VehicleRecord] = {}
        if vehicles:
            for vehicle in vehicles:
                self.vehicles[vehicle.vehicle_id] = vehicle
        self.detail_count = max(0, min(detail_count, len(self.vehicles)))
        self._refresh_tracking()

    def _refresh_tracking(self) -> None:
        vehicle_ids = list(self.vehicles.keys())[: self.detail_count]
        for idx, vehicle in enumerate(self.vehicles.values()):
            vehicle.tracked = vehicle.vehicle_id in vehicle_ids

    def register_vehicle(self, vehicle: VehicleRecord) -> None:
        self.vehicles[vehicle.vehicle_id] = vehicle
        self._refresh_tracking()
        update_metric("vehicle_count", len(self.vehicles))

    def get_vehicle(self, vehicle_id: str) -> VehicleRecord | None:
        return self.vehicles.get(vehicle_id)

    def list_vehicles(self) -> list[VehicleRecord]:
        return list(self.vehicles.values())

    def list_tracked_vehicles(self) -> list[VehicleRecord]:
        return [vehicle for vehicle in self.vehicles.values() if vehicle.tracked]

    def generate_fleet(
        self,
        count: int = 1000,
        detail_count: int = 10,
        random_seed: int = 42,
        sources: list[str] | None = None,
        destinations: list[str] | None = None,
        capacity_choices: list[float] | None = None,
        consumption_choices: list[float] | None = None,
    ) -> list[VehicleRecord]:
        if count <= 0:
            return []
        sources = sources or [f"node_{idx}" for idx in range(1, 21)]
        destinations = destinations or [f"node_{idx}" for idx in range(21, 41)]
        capacity_choices = capacity_choices or [30.0, 45.0, 60.0]
        consumption_choices = consumption_choices or [180.0, 200.0, 240.0]

        random.seed(int(random_seed))
        self.vehicles.clear()
        for idx in range(1, count + 1):
            vehicle_id = f"ev_{idx}"
            source = random.choice(sources)
            destination = random.choice(destinations)
            while destination == source:
                destination = random.choice(destinations)

            battery_capacity_kwh = random.choice(capacity_choices)
            consumption_wh_per_km = random.choice(consumption_choices)
            battery_pct = float(random.randint(5, 95))
            remaining = remaining_range_km(battery_pct, battery_capacity_kwh, consumption_wh_per_km)
            charging_requirement = energy_needed_kwh(battery_pct, 100.0, battery_capacity_kwh)

            self.vehicles[vehicle_id] = VehicleRecord(
                vehicle_id=vehicle_id,
                source=source,
                destination=destination,
                battery_capacity_kwh=round(battery_capacity_kwh, 1),
                battery_pct=battery_pct,
                remaining_range_km=round(remaining, 2),
                charging_requirement_kwh=round(charging_requirement, 2),
            )

        self.detail_count = max(0, min(detail_count, count))
        # Keep tracked EVs distinct in initial SoC so dashboard/recommendation
        # comparisons are meaningful during demo runs.
        if self.detail_count > 0:
            tracked_ids = list(self.vehicles.keys())[: self.detail_count]
            used_soc: set[int] = set()
            for vehicle_id in tracked_ids:
                vehicle = self.vehicles[vehicle_id]
                rounded_soc = int(round(vehicle.battery_pct))
                if rounded_soc in used_soc:
                    for candidate in range(5, 96):
                        if candidate not in used_soc:
                            vehicle.battery_pct = float(candidate)
                            vehicle.charging_requirement_kwh = round(energy_needed_kwh(vehicle.battery_pct, 100.0, vehicle.battery_capacity_kwh), 2)
                            break
                used_soc.add(int(round(vehicle.battery_pct)))
        self._refresh_tracking()
        update_metric("vehicle_count", len(self.vehicles))
        return self.list_vehicles()

    def save_manifest(self, manifest_path: str | Path) -> None:
        manifest_file = Path(manifest_path)
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        vehicles = [vehicle.to_dict() for vehicle in self.list_vehicles()]
        manifest_file.write_text(json.dumps(vehicles, indent=2), encoding="utf-8")

    def get_fleet_summary(self) -> dict[str, Any]:
        total = len(self.vehicles)
        return {
            "total_vehicles": total,
            "tracked_vehicles": len(self.list_tracked_vehicles()),
            "min_battery_pct": min((v.battery_pct for v in self.vehicles.values()), default=0.0),
            "max_battery_pct": max((v.battery_pct for v in self.vehicles.values()), default=0.0),
        }
