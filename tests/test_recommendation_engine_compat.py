import sys
import types

from src.recommendation_engine import engine as recommendation_engine_module
from src.recommendation_engine.engine import RecommendationEngine
from src.station_management.station_state import ChargingStation, StationManager


def test_recommend_supports_legacy_signature_without_model() -> None:
    stations = [
        ChargingStation("A", capacity=2, occupancy=0, waiting_time=4.0),
        ChargingStation("C", capacity=2, occupancy=0, waiting_time=1.0),
        ChargingStation("D", capacity=1, occupancy=0, waiting_time=2.0),
    ]
    station_manager = StationManager(stations)
    engine = RecommendationEngine(station_manager)

    result = engine.recommend("A", "D")

    assert result["station"] in {"A", "C", "D"}
    assert result["recommendation_reason"].startswith("heuristic")
    assert result["travel_distance"] >= 0.0


def test_recommendation_engine_loads_ppo_once_and_reuses_it(monkeypatch, tmp_path) -> None:
    stations = [ChargingStation("A", capacity=2, occupancy=0, waiting_time=1.0)]
    station_manager = StationManager(stations)
    model_path = tmp_path / "ppo_ev_final.zip"
    model_path.write_bytes(b"dummy model file")
    station_manager.get_station_state = lambda station_id: {"station_id": station_id, "ports": [{"power_kw": 22.0, "status": "Available"}]}

    load_calls = {"count": 0}

    class DummyModel:
        def predict(self, obs, deterministic=True):
            return 0, None

    def fake_load(model_path):
        load_calls["count"] += 1
        return DummyModel()

    class DummyEnv:
        vehicle_feat_dim = 6

        def __init__(self, *args, **kwargs):
            self.current_candidates = [{"station_id": "A", "distance_km": 1.0, "travel_time_min": 2.0, "avg_wait_min": 1.0, "queue_len": 0, "free_ports": 2, "price_per_kwh": 0.2, "grid_load_kw": 0.0}]

        def reset(self):
            return ({"vehicles": __import__("numpy").zeros((1, 6)), "stations": __import__("numpy").zeros((1, 6))}, {})

        def step(self, action):
            return ({}, 1.0, False, False, {})

        def close(self):
            return None

    dummy_module = types.ModuleType("src.rl_env.gym_ev_charging_env")
    dummy_module.GymEVChargingEnv = DummyEnv
    monkeypatch.setitem(sys.modules, "src.rl_env.gym_ev_charging_env", dummy_module)
    monkeypatch.setattr(recommendation_engine_module.PPO, "load", fake_load, raising=False)

    engine = RecommendationEngine(station_manager, model_path=str(model_path))

    first = engine.recommend({"battery_pct": 40.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 120.0})
    second = engine.recommend({"battery_pct": 42.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 118.0})

    assert load_calls["count"] == 1
    assert engine.model_load_count == 1
    assert first["station"] == "A"
    assert second["station"] == "A"
    assert first["recommendation_reason"].startswith("ppo_model:")
