from __future__ import annotations

import json
from pathlib import Path

from src.visualization.services.visualization_builder import VisualizationSnapshotBuilder
from src.xai.explainer import RecommendationExplainer


class DummyStationManager:
    def __init__(self) -> None:
        self._stations = [
            {"station_id": "S1", "lat": 12.97, "lon": 77.59},
            {"station_id": "S2", "lat": 12.98, "lon": 77.60},
            {"station_id": "S3", "lat": 12.99, "lon": 77.61},
        ]

    def list_all_stations(self) -> list[dict]:
        return self._stations

    def get_station_metrics(self, station_id: str) -> dict:
        mapping = {
            "S1": {"available_ports": 3, "occupied_ports": 1, "total_ports": 4, "price_per_kwh": 0.22, "avg_wait_estimate": 2.0, "power_kw": 50.0},
            "S2": {"available_ports": 2, "occupied_ports": 2, "total_ports": 4, "price_per_kwh": 0.28, "avg_wait_estimate": 5.0, "power_kw": 40.0},
            "S3": {"available_ports": 1, "occupied_ports": 3, "total_ports": 4, "price_per_kwh": 0.30, "avg_wait_estimate": 8.0, "power_kw": 35.0},
        }
        return mapping[station_id]

    def get_station_state(self, station_id: str) -> dict:
        for station in self._stations:
            if station["station_id"] == station_id:
                return station
        return {}


class DummyVehicleManager:
    def list_vehicles(self) -> list[object]:
        return []

    def list_tracked_vehicles(self) -> list[object]:
        return []


class DummyController:
    def __init__(self) -> None:
        self.station_manager = DummyStationManager()
        self.vehicle_manager = DummyVehicleManager()
        self.assignments = {}
        self.recommendation_log = []
        self.charging_events = []
        self.simulation_metrics = []
        self.last_explanation = None


def test_explainer_generates_complete_payload() -> None:
    explainer = RecommendationExplainer()
    vehicle_state = {"battery_pct": 18.0, "battery_capacity_kwh": 70.0, "remaining_range_km": 25.0}
    recommendation = {
        "station": "S1",
        "travel_distance": 3.5,
        "waiting_time": 2.0,
        "queue_length": 1,
        "charging_cost": 9.0,
        "available_ports": 3,
        "grid_load": 50.0,
        "ppo_reward": 0.73,
    }

    explanation = explainer.explain_recommendation(
        vehicle_state=vehicle_state,
        recommendation=recommendation,
        station_manager=DummyStationManager(),
    )

    assert explanation["decision_explanation"]
    assert explanation["confidence_score"] >= 0.0
    assert explanation["feature_importance"]["battery"] >= 0.0
    assert explanation["reward_decomposition"]["total_reward"] >= 0.0
    assert explanation["top_k_station_ranking"][0]["station_id"] == "S1"
    assert explanation["policy_probability_visualization"]["selected_station_id"] == "S1"
    assert explanation["action_probability_chart"][0]["station_id"] == "S1"


def test_export_explanation_to_json(tmp_path: Path) -> None:
    explainer = RecommendationExplainer()
    payload = explainer.explain_recommendation(
        vehicle_state={"battery_pct": 25.0, "battery_capacity_kwh": 60.0, "remaining_range_km": 35.0},
        recommendation={"station": "S2", "travel_distance": 2.0, "waiting_time": 4.0, "queue_length": 2, "charging_cost": 8.0, "available_ports": 2, "grid_load": 40.0},
        station_manager=DummyStationManager(),
    )

    exported = explainer.export_explanation_json(payload, tmp_path / "explanation.json")
    assert exported.exists()
    data = json.loads(exported.read_text(encoding="utf-8"))
    assert data["decision_explanation"] == payload["decision_explanation"]


def test_visualization_builder_includes_explanation_payload() -> None:
    controller = DummyController()
    controller.last_explanation = {
        "decision_explanation": "Low battery and short queue made S1 attractive.",
        "confidence_score": 0.88,
    }
    builder = VisualizationSnapshotBuilder(controller)

    payload = builder.build()

    assert payload["xai"]["latest_explanation"]["confidence_score"] == 0.88
