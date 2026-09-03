import json

from src.dashboard.dashboard import Dashboard


def test_dashboard_shutdown_writes_completed_disconnected_final_state(tmp_path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=1, station_count=1, tracked=1)

    dashboard.update({
        "simulation": {
            "step": 7200,
            "time": 7200.0,
            "status": "running",
            "running": True,
            "connection": "LIVE",
        },
        "vehicles": [],
        "stations": [],
        "tracked_ids": [],
        "metrics": [],
    })

    dashboard.finalize({
        "simulation": {
            "step": 7200,
            "time": 7200.0,
            "status": "COMPLETED",
            "simulation_status": "FINISHED",
            "running": False,
            "connection": "DISCONNECTED",
            "connection_status": "DISCONNECTED",
        },
        "vehicles": [],
        "stations": [],
        "tracked_ids": [],
        "metrics": [],
    })

    payload = json.loads((tmp_path / "dashboard_state.json").read_text(encoding="utf-8"))
    simulation = payload["simulation"]
    assert simulation["step"] == 7200
    assert simulation["simulation_status"] == "COMPLETED"
    assert simulation["status"] == "completed"
    assert simulation["connection_status"] == "DISCONNECTED"
    assert simulation["connection"] == "DISCONNECTED"
    assert simulation["running"] is False