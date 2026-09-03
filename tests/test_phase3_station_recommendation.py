from __future__ import annotations

import types

from src.recommendation_engine.engine import RecommendationEngine
from src.simulation.controller import SimulationController


class StubStationManager:
    def __init__(self) -> None:
        self._stations = [
            {"station_id": "station-001", "name": "Alpha", "lat": 12.97, "lon": 77.59},
            {"station_id": "station-002", "name": "Beta", "lat": 12.98, "lon": 77.60},
        ]

    def list_all_stations(self):
        return list(self._stations)

    def get_station_metrics(self, station_id):
        if station_id == "station-001":
            return {"station_id": "station-001", "available_ports": 1, "total_ports": 2, "price_per_kwh": 0.2, "avg_wait_estimate": 4.0, "power_kw": 22.0, "queue_length": 1, "grid_load_kw": 10.0, "name": "Alpha"}
        if station_id == "station-002":
            return {"station_id": "station-002", "available_ports": 2, "total_ports": 3, "price_per_kwh": 0.25, "avg_wait_estimate": 2.0, "power_kw": 22.0, "queue_length": 0, "grid_load_kw": 6.0, "name": "Beta"}
        return None

    def get_station_state(self, station_id):
        metrics = self.get_station_metrics(station_id)
        if metrics is None:
            return None
        return {"station_id": station_id, "ports": [{"port_id": f"{station_id}-p1", "power_kw": metrics["power_kw"], "status": "Available"}]}

    def resolve_station_id(self, station_id):
        if station_id in {s["station_id"] for s in self._stations}:
            return station_id
        lookup = {s["station_id"].lower(): s["station_id"] for s in self._stations}
        return lookup.get(str(station_id).strip().lower())


class FakePPOModel:
    def __init__(self, action: int = 1) -> None:
        self.action = action

    def predict(self, obs, deterministic=True):
        return self.action, {"deterministic": deterministic}


class FakePPO:
    @classmethod
    def load(cls, path):
        return FakePPOModel(action=1)


def test_recommendation_engine_uses_database_station_ids(monkeypatch):
    monkeypatch.setattr("src.recommendation_engine.engine.PPO", FakePPO)

    engine = RecommendationEngine(station_manager=StubStationManager(), model_path="fake.zip")
    result = engine.recommend(
        {"battery_pct": 35.0, "battery_capacity_kwh": 60.0, "remaining_range_km": 70.0},
        model_path="fake.zip",
        candidate_count=2,
        tracked_vehicle_count=10,
    )

    assert result["station"] == "station-002"
    assert result["station_name"] == "Beta"
    assert result["recommendation_score"] is not None


def test_recommendation_engine_falls_back_to_valid_station_for_invalid_selection(monkeypatch):
    monkeypatch.setattr("src.recommendation_engine.engine.PPO", FakePPO)

    class InvalidStationEnv:
        def __init__(self, *args, **kwargs):
            self.current_candidates = [{"station_id": "missing-station", "distance_km": 1.0, "travel_time_min": 2.0, "free_ports": 0, "total_ports": 1, "price_per_kwh": 0.0, "avg_wait_min": 2.0, "queue_len": 0, "grid_load_kw": 0.0}]

        def reset(self):
            return {"vehicles": None, "stations": None}, {}

        def step(self, action):
            return {"vehicles": None, "stations": None}, 0.0, False, False, {"chosen_station": "missing-station"}

        def close(self):
            return None

    import src.recommendation_engine.engine as engine_module

    monkeypatch.setattr(engine_module, "GymEVChargingEnv", InvalidStationEnv)

    engine = RecommendationEngine(station_manager=StubStationManager(), model_path="fake.zip")
    result = engine.recommend(
        {"battery_pct": 35.0, "battery_capacity_kwh": 60.0, "remaining_range_km": 70.0},
        model_path="fake.zip",
        candidate_count=2,
        tracked_vehicle_count=10,
    )

    assert result["station"] == "station-001"
    assert result["station_name"] == "Alpha"


def test_controller_recommendation_entry_captures_required_telemetry_fields():
    controller = SimulationController.__new__(SimulationController)
    controller.station_manager = StubStationManager()
    controller.net = None
    controller.recommendation_log = []
    controller.charging_events = []
    controller.explainer = None
    controller.last_explanation = None
    controller.reward_curve = []

    rec = {
        "station": "station-002",
        "station_name": "Beta",
        "travel_distance": 2.5,
        "travel_time": 3.5,
        "waiting_time": 1.5,
        "queue_length": 0,
        "charging_cost": 1.25,
        "grid_load": 6.0,
        "available_ports": 2,
        "action": 1,
        "recommendation_score": 4.5,
        "recommendation_reason": "ppo_model",
    }
    station_metrics = controller.station_manager.get_station_metrics("station-002")

    entry = controller._build_recommendation_log_entry(
        step=7,
        vehicle_id="ev-1",
        vehicle_state={"battery_pct": 20.0, "battery_capacity_kwh": 60.0, "remaining_range_km": 40.0},
        station_id="station-002",
        station_metrics=station_metrics,
        charging_price_per_kwh=0.25,
        estimated_session_cost=2.5,
        rec=rec,
        ppo_reward=4.5,
    )

    assert entry["selected_station"] == "station-002"
    assert entry["selected_station_name"] == "Beta"
    assert entry["distance_km"] == 2.5
    assert entry["price_per_kwh"] == 0.25
    assert entry["available_ports"] == 2
    assert entry["queue_length"] == 0
    assert entry["estimated_wait_min"] == 1.5
    assert entry["station_load_kw"] == 6.0
    assert entry["recommendation_score"] == 4.5
    assert entry["ppo_action"] == 1
    assert entry["recommendation_timestamp_utc"]


def test_controller_route_validation_blocks_unreachable_station(monkeypatch):
    controller = SimulationController.__new__(SimulationController)
    controller.net = None
    controller.station_manager = StubStationManager()

    class FakeTraCI:
        @staticmethod
        def vehicle(*args, **kwargs):
            raise AssertionError("should not call vehicle APIs")

    monkeypatch.setattr("src.simulation.controller.traci", FakeTraCI)

    valid, reason = controller._validate_station_route("ev-1", "station-999", target_edge_id="edge-1")
    assert valid is False
    assert "station" in reason.lower()
