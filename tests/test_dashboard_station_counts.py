import json

from src.dashboard.dashboard import Dashboard


def test_dashboard_station_counts_distinguish_osm_registered_and_active(tmp_path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=1, station_count=3, tracked=1)
    dashboard.update({
        "simulation": {"step": 1, "running": True},
        "station_details": [
            {"station_id": "osm_1", "name": "OSM One", "total_ports": 4, "available_ports": 4, "occupied_ports": 0, "queue_length": 0},
            {"station_id": "osm_2", "name": "OSM Two", "total_ports": 4, "available_ports": 3, "occupied_ports": 1, "queue_length": 0},
            {"station_id": "registered_1", "name": "Registered One", "total_ports": 2, "available_ports": 2, "occupied_ports": 0, "queue_length": 0},
        ],
        "vehicles": [],
    })

    payload = json.loads((tmp_path / "dashboard_state.json").read_text(encoding="utf-8"))
    assert payload["station_statistics"] == {"real_osm": 2, "registered": 3, "active": 1}
    assert len(payload["stations"]) == 3