from src.ev_management.vehicle_manager import VehicleRecord
from src.simulation.controller import SimulationController


def _vehicle(battery_pct: float, remaining_range_km: float) -> VehicleRecord:
    return VehicleRecord(
        vehicle_id="ev-test",
        source="node_a",
        destination="node_b",
        battery_capacity_kwh=60.0,
        battery_pct=battery_pct,
        remaining_range_km=remaining_range_km,
        charging_requirement_kwh=0.0,
        tracked=True,
    )


def test_high_soc_vehicle_does_not_trigger_charging_route() -> None:
    controller = SimulationController.__new__(SimulationController)
    controller.charge_threshold_pct = 20.0

    assert controller._should_seek_charger(_vehicle(73.0, 120.0)) is False


def test_low_soc_vehicle_triggers_charging_route() -> None:
    controller = SimulationController.__new__(SimulationController)
    controller.charge_threshold_pct = 20.0

    assert controller._should_seek_charger(_vehicle(11.31, 15.0)) is True