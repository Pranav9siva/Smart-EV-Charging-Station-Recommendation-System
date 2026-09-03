from __future__ import annotations

from src.analytics.reporting import AnalyticsEngine
from src.evaluation.benchmark import BenchmarkRunner
from src.persistence.repository import Repository
from src.mlops.registry import ModelRegistry


def test_evaluator_writes_outputs(tmp_path) -> None:
    runner = BenchmarkRunner(tmp_path)
    report = runner.run([
        {"selected_station": "S1", "ppo_reward": 0.8, "queue_length": 2, "waiting_time": 1.5, "charging_cost": 7.0, "energy_consumption": 15.0},
        {"selected_station": "S2", "ppo_reward": 0.5, "queue_length": 4, "waiting_time": 3.0, "charging_cost": 10.0, "energy_consumption": 20.0},
    ])
    assert report["average_reward"] >= 0.0
    assert (tmp_path / "evaluation_report.json").exists()


def test_analytics_engine_generates_summary(tmp_path) -> None:
    engine = AnalyticsEngine(tmp_path)
    summary = engine.generate_reports([
        {"energy_consumption": 10.0, "charging_cost": 6.0},
        {"energy_consumption": 20.0, "charging_cost": 7.0},
    ])
    assert summary["energy_consumption"] == 30.0
    assert summary["revenue_estimation"] == 13.0


def test_repository_persists_data(tmp_path) -> None:
    repo = Repository(tmp_path / "test.db")
    repo.save_recommendation({"selected_station": "S1", "ppo_reward": 0.2})
    repo.save_vehicle_history("V1", {"battery_pct": 20.0})
    repo.save_charging_session("S1", "V1", {"status": "charging"})


def test_model_registry_registers_metadata(tmp_path) -> None:
    registry = ModelRegistry(tmp_path)
    path = registry.register("ppo", "v1", {"accuracy": 0.9})
    assert path.exists()
