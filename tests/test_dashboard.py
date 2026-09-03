import json

from src.dashboard.dashboard import Dashboard


def test_dashboard_writes_live_state_and_keeps_visible_data(tmp_path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=1000, station_count=2, tracked=10)

    state = {
        "vehicles": [
            {
                "id": f"ev_{index + 1}",
                "battery_pct": 50 + index,
                "recommended_station": "st_1" if index == 0 else None,
                "charging_status": "charging" if index == 0 else "driving",
                "current_position": {"x": float(index), "y": float(index + 1)},
            }
            for index in range(10)
        ],
        "stations": [
            {
                "station_id": "st_1",
                "available_ports": 2,
                "occupied_ports": 1,
                "grid_load_kw": 22.0,
            }
        ],
        "tracked_ids": [f"ev_{index + 1}" for index in range(10)],
        "summary": {"charging_vehicles": 1, "avg_wait_min": 3.2},
        "metrics": [{"step": 1, "charging": 1, "avg_wait_min": 3.2}],
        "charging_events": [{"event_type": "charging_completed", "step": 9, "vehicle_id": "ev_1"}],
    }

    dashboard.update(state)

    payload = json.loads((tmp_path / "dashboard_state.json").read_text(encoding="utf-8"))
    assert len(payload["vehicles"]) == 1000
    assert len(payload["tracked_ids"]) == 10
    assert payload["summary"]["charging_vehicles"] == 1
    assert payload["summary"]["avg_wait_min"] == 3.2
    assert payload["charging_events"][0]["event_type"] == "charging_completed"
    assert payload["simulation"]["connection_status"] == "LIVE"
    assert payload["simulation"]["simulation_status"] == "RUNNING"
    assert (tmp_path / "research_dashboard.html").exists()
    html = (tmp_path / "research_dashboard.html").read_text(encoding="utf-8")
    assert "EV Charging Research Dashboard" in html
