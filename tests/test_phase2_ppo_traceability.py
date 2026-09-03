import importlib
from types import SimpleNamespace

from src.dashboard.dashboard import Dashboard
from src.recommendation_engine import engine as recommendation_engine_module


class StubStationManager:
    def __init__(self) -> None:
        self._stations = [
            {"station_id": "st-1", "name": "Alpha", "lat": 12.97, "lon": 77.59},
            {"station_id": "st-2", "name": "Beta", "lat": 12.98, "lon": 77.60},
        ]

    def list_all_stations(self):
        return list(self._stations)

    def get_station_state(self, station_id):
        if station_id == "st-1":
            return {"station_id": "st-1", "ports": [{"port_id": "p1", "power_kw": 22.0, "status": "Available"}]}
        if station_id == "st-2":
            return {"station_id": "st-2", "ports": [{"port_id": "p2", "power_kw": 22.0, "status": "Available"}]}
        return None

    def get_station_metrics(self, station_id):
        if station_id == "st-1":
            return {"station_id": "st-1", "available_ports": 1, "total_ports": 2, "price_per_kwh": 0.2, "avg_wait_estimate": 4.0, "power_kw": 22.0}
        if station_id == "st-2":
            return {"station_id": "st-2", "available_ports": 2, "total_ports": 3, "price_per_kwh": 0.25, "avg_wait_estimate": 2.0, "power_kw": 22.0}
        return None


class FakePPOModel:
    def __init__(self, *, action: int = 1) -> None:
        self.action = action
        self.calls = []

    def predict(self, obs, deterministic=True):
        self.calls.append({"vehicles_shape": obs["vehicles"].shape, "stations_shape": obs["stations"].shape, "deterministic": deterministic})
        return self.action, {"deterministic": deterministic}


class FakePPO:
    @classmethod
    def load(cls, path):
        return FakePPOModel(action=1)


def test_recommendation_engine_uses_real_ppo_inference_and_exposes_trace(monkeypatch):
    monkeypatch.setattr(recommendation_engine_module, "PPO", FakePPO)

    engine = recommendation_engine_module.RecommendationEngine(station_manager=StubStationManager(), model_path="fake.zip")
    result = engine.recommend(
        {"battery_pct": 35.0, "battery_capacity_kwh": 60.0, "remaining_range_km": 70.0},
        model_path="fake.zip",
        candidate_count=2,
        tracked_vehicle_count=10,
    )

    assert result["station"] in {"st-1", "st-2"}
    assert result["action"] == 1
    assert result["ppo_reward"] is not None
    assert result["recommendation_reason"].startswith("ppo_model")
    assert result["observation"]["vehicles_shape"] == [10, 6]
    assert result["observation"]["stations_shape"] == [2, 6]


def test_recommendation_engine_maps_action_index_to_real_station(monkeypatch):
    monkeypatch.setattr(recommendation_engine_module, "PPO", FakePPO)

    engine = recommendation_engine_module.RecommendationEngine(station_manager=StubStationManager(), model_path="fake.zip")
    result = engine.recommend(
        {"battery_pct": 40.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 100.0},
        model_path="fake.zip",
        candidate_count=2,
        tracked_vehicle_count=10,
    )

    assert result["station"] in {"st-1", "st-2"}
    assert result["selected_station"] == result["station"]


def test_dashboard_canonical_payload_preserves_ppo_reason_and_history(monkeypatch):
    dashboard = Dashboard(output_dir="/tmp/ppo-dashboard", fleet_size=1, station_count=1, tracked=1)
    state = {
        "simulation": {"step": 4, "time": 4.0, "status": "running", "connection": "LIVE"},
        "ppo_decision": {"ev": "ev-1", "station": "st-1", "action": 0, "reward": 3.2, "battery": 72.0, "reason": "lowest_wait", "timestamp": 4},
        "history": {"rewards": [3.2], "recommendations": [{"vehicle_id": "ev-1", "selected_station": "st-1"}], "timestamps": [4]},
        "tracked_ev": {"id": "ev-1", "selected_station": "st-1"},
        "vehicles": [{"id": "ev-1", "x": 10.0, "y": 20.0, "speed": 4.5, "battery": 72.0, "state": "charging"}],
        "station_details": [{"id": "st-1", "name": "Alpha", "x": 100.0, "y": 200.0, "total_ports": 4, "available_ports": 2, "queue": 1, "price": 0.22, "wait": 2.0, "load": 40.0, "utilization": 0.5}],
        "traffic": [],
        "ppo": [],
        "tracked_ids": ["ev-1"],
        "metrics": [],
        "recommendations": [],
        "charging_events": [],
        "latest_recommendation": None,
        "network": {},
        "xai": {},
        "kpis": {},
        "stations": [{"id": "st-1"}],
        "stations_summary": {"total": 1, "available": 1, "occupied": 0, "queue_total": 0},
    }

    payload = dashboard._build_canonical_state(state)

    assert payload["ppo_decision"]["reason"] == "lowest_wait"
    assert payload["history"]["rewards"] == [3.2]
    assert payload["tracked_ev"]["selected_station"] == "st-1"
