from __future__ import annotations

from src.dashboard.dashboard import Dashboard


def test_dashboard_chart_payload_uses_live_ppo_and_station_metrics(tmp_path):
    dashboard = Dashboard(output_dir=str(tmp_path))
    state = {
        "simulation": {"step": 3, "time": 3, "speed": 1.2, "ev_count": 2, "running": True},
        "vehicles": [],
        "station_details": [
            {"station_id": "S1", "name": "North Hub", "total_ports": 8, "available_ports": 4, "queue": 2, "utilization": 0.5},
            {"station_id": "S2", "name": "West Hub", "total_ports": 4, "available_ports": 1, "queue": 3, "utilization": 0.75},
        ],
        "ppo": [
            {"vehicle_id": "EV1", "action": "charge", "station_id": "S1", "reward": 1.25, "decision_time": 2, "battery": 80},
            {"vehicle_id": "EV2", "action": "wait", "station_id": "S2", "reward": -0.4, "decision_time": 3, "battery": 30},
        ],
        "recommendations": [
            {"vehicle_id": "EV3", "selected_station": "S2", "ppo_reward": 0.65, "step": 1},
        ],
        "tracked_ids": ["EV2"],
        "kpis": {"avg_speed": 2.0, "energy_consumed_kwh": 3.5},
    }

    canonical = dashboard._build_canonical_state(state)

    charts = canonical["charts"]
    assert charts["reward_series"][0]["reward"] == 1.25
    assert charts["reward_series"][1]["reward"] == -0.4
    assert charts["reward_series"][2]["reward"] == 0.65
    assert charts["station_series"][0]["id"] == "S1"
    assert charts["station_series"][0]["utilization"] == 0.5
    assert charts["station_series"][0]["available_ports"] == 4
    assert charts["station_series"][0]["queue"] == 2


def test_dashboard_chart_payload_ignores_invalid_values(tmp_path):
    dashboard = Dashboard(output_dir=str(tmp_path))
    state = {
        "simulation": {"step": 1, "time": 1, "speed": 1.0, "ev_count": 1, "running": True},
        "vehicles": [],
        "station_details": [
            {"station_id": "S1", "name": "North Hub", "total_ports": 4, "available_ports": "bad", "queue": None, "utilization": "NaN"},
        ],
        "ppo": [
            {"vehicle_id": "EV1", "action": "charge", "station_id": "S1", "reward": "nan", "decision_time": 1, "battery": 80},
        ],
        "recommendations": [],
        "tracked_ids": ["EV1"],
        "kpis": {},
    }

    canonical = dashboard._build_canonical_state(state)

    charts = canonical["charts"]
    assert charts["reward_series"] == []
    assert charts["station_series"][0]["available_ports"] is None
    assert charts["station_series"][0]["queue"] is None
    assert charts["station_series"][0]["utilization"] is None
