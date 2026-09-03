import json

from src.dashboard.dashboard import Dashboard


def test_dashboard_exposes_action_for_multiple_recommendations(tmp_path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=2, station_count=2, tracked=2)

    dashboard.update({
        "simulation": {"step": 12, "running": True},
        "vehicles": [],
        "stations": [],
        "ppo_decision": {
            "ev_id": "ev_1",
            "selected_station": "station_a",
            "reward": 4.5,
            "battery": 18.0,
            "reason": "weighted_multi_factor_score",
        },
        "recommendations": [
            {"vehicle_id": "ev_1", "selected_station": "station_a", "ppo_reward": 4.5, "battery_pct": 18.0},
            {"vehicle_id": "ev_2", "selected_station": "station_b", "ppo_reward": 3.2, "battery_pct": 16.0},
        ],
    })

    payload = json.loads((tmp_path / "dashboard_state.json").read_text(encoding="utf-8"))
    decision = payload["ppo_decision"]
    assert decision["action"] == "station_a"
    assert decision["selected_station"] == "station_a"
    assert decision["reward"] == 4.5
    assert decision["battery"] == 18.0
    assert decision["reason"] == "weighted_multi_factor_score"

    dashboard.update({
        "simulation": {"step": 13, "running": True},
        "vehicles": [],
        "stations": [],
        "ppo_decision": {
            "ev_id": "ev_2",
            "selected_station": "station_b",
            "ppo_action": 0,
            "reward": 3.2,
            "battery": 16.0,
            "reason": "lower_wait",
        },
    })

    payload = json.loads((tmp_path / "dashboard_state.json").read_text(encoding="utf-8"))
    decision = payload["ppo_decision"]
    assert decision["action"] == 0
    assert decision["selected_station"] == "station_b"
    assert decision["reward"] == 3.2