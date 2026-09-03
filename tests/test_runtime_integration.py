from __future__ import annotations

from src.api.router import explain, recommend
from src.xai.explainer import RecommendationExplainer


def test_recommendation_explainer_enriches_payload() -> None:
    explainer = RecommendationExplainer()
    explanation = explainer.explain_recommendation(
        vehicle_state={"battery_pct": 30.0, "remaining_range_km": 12.0, "battery_capacity_kwh": 70.0},
        recommendation={"station": "S1", "travel_distance": 3.0, "waiting_time": 1.5, "queue_length": 1, "charging_cost": 5.0, "available_ports": 2, "grid_load": 20.0, "ppo_reward": 0.3},
        station_manager=None,
    )
    assert "shap_explanation" in explanation
    assert "lime_explanation" in explanation
    assert "counterfactual_explanation" in explanation


def test_api_explain_endpoint_uses_request_context() -> None:
    payload = explain(type("Req", (), {"station_id": "S2", "battery_pct": 40.0})())
    assert payload["station_id"] == "S2"
    assert "decision_explanation" in payload
