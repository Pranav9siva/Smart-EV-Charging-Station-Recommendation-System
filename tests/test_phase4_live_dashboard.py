from __future__ import annotations

from types import SimpleNamespace

from src.simulation.controller import SimulationController


class FakeVehicleManager:
    def __init__(self) -> None:
        self.vehicles = {}

    def list_vehicles(self):
        return []

    def generate_fleet(self, count: int, detail_count: int = 0):
        return []


class FakeRuntimeControl:
    def read(self):
        return {"status": "play", "speed": 1.0, "step_once": False, "reset": False}

    def consume_reset(self):
        return None

    def consume_step(self):
        return None


class FakeTraCI:
    class simulation:
        step = 0

        @staticmethod
        def Step():
            FakeTraCI.simulation.step += 1

        @staticmethod
        def getTime():
            return FakeTraCI.simulation.step

    @staticmethod
    def simulationStep():
        FakeTraCI.simulation.step += 1

    class vehicle:
        @staticmethod
        def getIDList():
            return []


def test_run_refreshes_dashboard_on_every_step(monkeypatch):
    controller = SimulationController.__new__(SimulationController)
    controller.fleet_size = 0
    controller.station_count = 0
    controller.tracked = 0
    controller.charge_threshold_pct = 20.0
    controller.dashboard_state_interval = 5
    controller.visualization_interval = 999
    controller.metrics_interval = 999
    controller.dashboard = SimpleNamespace(update=lambda state: None, finalize=lambda state: None)
    controller.vehicle_manager = FakeVehicleManager()
    controller.station_manager = SimpleNamespace()
    controller.recommender = SimpleNamespace()
    controller.assignments = {}
    controller.recommendation_log = []
    controller.charging_events = []
    controller.tracked_lifecycle = {}
    controller.simulation_metrics = []
    controller.reward_curve = []
    controller.last_explanation = None
    controller.runtime_control = FakeRuntimeControl()
    controller.vm_to_sumo = {}
    controller.sumo_to_vm = {}
    controller.last_positions = {}
    controller.injected_count = 0
    controller.low_battery_count = 0
    controller.rec_call_count = 0
    controller._initial_battery_snapshot = {}
    controller._station_snapshot_cache = []
    controller._station_snapshot_cache_by_id = {}
    controller._station_snapshot_cache_step = None
    controller.current_step = 0
    controller._last_dashboard_state = None
    controller._monitoring_collectors = []
    controller._monitoring_started_at = 0.0
    controller._monitoring_last_step_time = 0.0
    controller._monitoring_step_count = 0
    controller._monitoring_episode = 0
    controller._monitoring_last_reward = 0.0
    controller._timing_totals_ms = {}
    controller._timing_max_ms = {}
    controller._timing_counts = {}
    controller._timing_started_at = 0.0
    controller._sumo_process = None
    controller.net = None
    controller.explainer = None
    controller.visualization = None
    controller._is_traci_connected = lambda: True
    controller.start = lambda: None
    controller.stop = lambda: None
    controller._invalidate_station_snapshots = lambda: None
    controller._get_known_sumo_vehicle_ids = lambda: set()
    controller._update_vehicles = lambda step: None
    controller._process_charging = lambda step: None
    controller._collect_metrics = lambda step: None
    controller._emit_runtime_metrics = lambda step, elapsed: None
    controller._save_results = lambda: None

    refresh_calls = []

    def fake_refresh_dashboard():
        refresh_calls.append(controller.current_step)

    controller._refresh_dashboard = fake_refresh_dashboard

    monkeypatch.setattr("src.simulation.controller.traci", FakeTraCI)
    monkeypatch.setattr("src.simulation.controller.checkBinary", lambda _name: "sumo")
    monkeypatch.setattr("src.simulation.controller.getFreeSocketPort", lambda: 5000)

    controller.run(steps=2)

    assert len(refresh_calls) == 2
