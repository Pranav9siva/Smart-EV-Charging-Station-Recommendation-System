from __future__ import annotations

import src.simulation.controller as controller_module
from src.ev_management.vehicle_manager import VehicleManager
from src.simulation.controller import SimulationController


def test_injects_multiple_missing_vehicles_and_skips_invalid_routes(monkeypatch):
    controller = SimulationController.__new__(SimulationController)
    controller.vehicle_manager = VehicleManager()
    controller.vehicle_manager.generate_fleet(count=3, detail_count=1)
    controller.vm_to_sumo = {}
    controller.sumo_to_vm = {}
    controller.assignments = {}
    controller.last_positions = {}
    controller.tracked_lifecycle = {}
    controller.charging_events = []
    controller.recommendation_log = []
    controller.injected_count = 0
    controller.low_battery_count = 0
    controller.rec_call_count = 0
    controller._injected_success = 0
    controller._inject_first_exception = None
    controller._invalid_vehicle_routes = []

    monkeypatch.setattr(controller, "_get_known_sumo_vehicle_ids", lambda: set())
    monkeypatch.setattr(controller_module.traci.route, "getIDList", lambda: ["route_a", "route_b"])
    monkeypatch.setattr(controller_module.traci.edge, "getIDList", lambda: ["edge_1", "edge_2"])
    monkeypatch.setattr(controller_module.traci.simulation, "findRoute", lambda source, target: type("Route", (), {"edges": ["edge_1", "edge_2"]})())
    monkeypatch.setattr(controller_module.traci.route, "add", lambda route_id, edges: None)

    added: list[tuple[str, str]] = []

    def fake_vehicle_add(vehicle_id: str, route_id: str, *args, **kwargs):
        added.append((vehicle_id, route_id))
        if vehicle_id == "ev_1":
            raise RuntimeError("invalid route")

    monkeypatch.setattr(controller_module.traci.vehicle, "add", fake_vehicle_add)

    controller._inject_vehicles_into_sumo()

    assert controller._injected_success == 2
    assert controller.vm_to_sumo.get("ev_2") == "ev_2"
    assert controller.vm_to_sumo.get("ev_3") == "ev_3"
    assert len(added) == 3
