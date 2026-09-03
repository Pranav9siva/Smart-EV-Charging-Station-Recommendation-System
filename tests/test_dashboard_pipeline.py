import json
from pathlib import Path

from src.dashboard.dashboard import Dashboard


def test_dashboard_emits_canonical_live_telemetry_sections(tmp_path: Path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=1, station_count=1, tracked=1)

    state = {
        "simulation": {
            "step": 8,
            "time": 8.0,
            "status": "running",
            "speed": 1.5,
            "vehicle_count": 1,
            "ev_count": 1,
            "connection": "LIVE",
        },
        "tracked_ev": {
            "id": "ev-1",
            "x": 10.0,
            "y": 20.0,
            "road_id": "edge_1",
            "speed": 4.5,
            "battery": 72.0,
            "battery_percent": 72.0,
            "state": "charging",
            "selected_station": "st-1",
            "destination": "edge_2",
            "charging": True,
        },
        "ppo_decision": {
            "ev": "ev-1",
            "station": "st-1",
            "action": "charge",
            "reward": 3.2,
            "battery": 72.0,
            "reason": "lowest_wait",
            "timestamp": 8,
        },
        "station": {
            "id": "st-1",
            "name": "North Station",
            "x": 100.0,
            "y": 200.0,
            "ports_total": 4,
            "ports_available": 2,
            "queue": 1,
            "price": 0.22,
            "wait": 2.0,
            "load": 40.0,
            "utilization": 0.5,
        },
        "vehicles": [{
            "id": "ev-1",
            "x": 10.0,
            "y": 20.0,
            "speed": 4.5,
            "battery": 72.0,
            "state": "charging",
        }],
        "stations": [{
            "id": "st-1",
            "x": 100.0,
            "y": 200.0,
            "ports_total": 4,
            "ports_available": 2,
            "queue": 1,
            "price": 0.22,
            "wait": 2.0,
            "load": 40.0,
            "utilization": 0.5,
        }],
        "history": {
            "rewards": [3.2],
            "recommendations": [{"vehicle_id": "ev-1", "selected_station": "st-1"}],
            "timestamps": [8],
        },
        "kpis": {},
        "station_details": [],
        "traffic": [],
        "ppo": [],
        "tracked_ids": ["ev-1"],
        "metrics": [],
        "recommendations": [],
        "charging_events": [],
        "latest_recommendation": None,
        "network": {},
        "xai": {},
    }

    dashboard.update(state)

    payload = json.loads((tmp_path / "dashboard_state.json").read_text(encoding="utf-8"))
    assert payload["simulation"]["step"] == 8
    assert payload["simulation"]["status"] == "running"
    assert payload["simulation"]["connection"] == "LIVE"
    assert payload["tracked_ev"]["id"] == "ev-1"
    assert payload["tracked_ev"]["road_id"] == "edge_1"
    assert payload["ppo_decision"]["reason"] == "lowest_wait"
    assert payload["station"]["id"] == "st-1"
    assert payload["vehicles"][0]["id"] == "ev-1"
    assert payload["stations"][0]["id"] == "st-1"
    assert payload["history"]["recommendations"][0]["vehicle_id"] == "ev-1"


def test_dashboard_emits_canonical_state_and_history(tmp_path: Path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=2, station_count=1, tracked=1)

    simulation_state = {
        "simulation": {"step": 3, "time": 12.5},
        "kpis": {"avg_speed": 12.5, "traffic_density": 0.8, "avg_wait_min": 4.2, "energy_consumed_kwh": 3.7},
        "stations": {"total": 1, "available": 2, "occupied": 1, "queue_total": 1},
        "vehicles": [
            {
                "id": "ev-1",
                "battery_pct": 78.0,
                "status": "charging",
                "current_edge": "edge_1",
                "current_position": {"x": 10.0, "y": 20.0},
                "recommended_station": "st-1",
                "charging_price_per_kwh": 0.22,
                "queue_length": 1,
                "waiting_time": 3.0,
                "available_ports": 2,
                "occupied_ports": 1,
                "grid_load": 40.0,
                "ppo_reward": 8.7,
                "simulation_step": 3,
            }
        ],
        "station_details": [
            {
                "station_id": "st-1",
                "name": "North Station",
                "lat": 12.97,
                "lon": 77.59,
                "total_ports": 4,
                "available_ports": 2,
                "occupied_ports": 2,
                "queue_length": 1,
                "price_per_kwh": 0.22,
                "avg_wait_estimate": 3.0,
                "grid_load_kw": 40.0,
            }
        ],
        "traffic": [{"edge_id": "edge_1", "density": 3, "congestion": 0.5, "speed": 12.5}],
        "ppo": [{"vehicle_id": "ev-1", "action": "charge", "station_id": "st-1", "reward": 8.7, "battery": 78.0, "decision_time": 3, "decision_features": [0.1, 0.2]}],
        "tracked_ids": ["ev-1"],
        "metrics": [{"step": 3, "total_vehicles": 1, "assigned": 1, "charging": 1, "waiting": 0, "avg_wait_min": 3.0}],
        "recommendations": [],
        "charging_events": [],
        "latest_recommendation": None,
        "network": {},
        "xai": {},
    }

    dashboard.update(simulation_state)

    state_path = tmp_path / "dashboard_state.json"
    history_path = tmp_path / "dashboard_history.json"

    assert state_path.exists()
    assert history_path.exists()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["schema_version"] == 1
    assert state["stations"][0]["station_id"] == "st-1"
    assert state["tracked_vehicles"][0]["vehicle_id"] == "ev-1"
    assert state["tracked_vehicles"][0]["battery"] == 78.0
    assert state["traffic"]["average_speed"] == 12.5
    assert state["traffic"]["congestion_percent"] == 50.0
    assert state["ppo"][0]["reward"] == 8.7
    assert state["validation"]["missing_fields"] == []
    assert state["simulation"]["step"] == 3
    assert state["simulation"]["sim_time"] == 12.5
    assert state["network"]["average_speed"] == 12.5
    assert state["tracked_ev"]["id"] == "ev-1"
    assert state["station"]["id"] == "st-1"
    assert state["ppo"][0]["ev_id"] == "ev-1"
    assert state["history"]["rewards"][0] == 8.7
    assert state["stations"][0]["id"] == "st-1"
    assert state["vehicles"][0]["id"] == "ev-1"

    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history["schema_version"] == 1
    assert history["entries"][0]["tracked_vehicles"][0]["vehicle_id"] == "ev-1"


def test_dashboard_uses_latest_recommendation_for_tracked_ev_and_reason(tmp_path: Path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=1, station_count=1, tracked=1)

    state = {
        "simulation": {"step": 4, "time": 9.0, "running": True, "speed": 1.0, "fleet_size": 1, "ev_count": 1, "tracked_count": 1},
        "kpis": {"avg_speed": 10.0, "traffic_density": 0.4, "avg_wait_min": 1.5, "energy_consumed_kwh": 2.0},
        "vehicles": [
            {
                "id": "ev-1",
                "battery_pct": 45.0,
                "status": "driving",
                "current_position": {"x": 3.0, "y": 4.0},
            }
        ],
        "station_details": [
            {
                "station_id": "st-1",
                "name": "North Station",
                "lat": 12.97,
                "lon": 77.59,
                "total_ports": 4,
                "available_ports": 2,
                "occupied_ports": 2,
                "queue_length": 1,
                "price_per_kwh": 0.22,
                "avg_wait_estimate": 3.0,
                "grid_load_kw": 40.0,
                "utilization": 0.5,
            }
        ],
        "traffic": [{"edge_id": "edge_1", "density": 1, "congestion": 0.2, "speed": 10.0}],
        "ppo": [],
        "tracked_ids": ["ev-1"],
        "metrics": [],
        "recommendations": [],
        "charging_events": [],
        "latest_recommendation": {
            "vehicle_id": "ev-1",
            "selected_station": "st-1",
            "selected_station_name": "North Station",
            "battery_pct": 45.0,
            "ppo_reward": 4.5,
            "recommendation_reason": "lowest_cost",
        },
        "network": {},
        "xai": {},
    }

    dashboard.update(state)

    payload = json.loads((tmp_path / "dashboard_state.json").read_text(encoding="utf-8"))
    assert payload["ppo_decision"]["selected_station"] == "st-1"
    assert payload["ppo_decision"]["reason"] == "lowest_cost"
    assert payload["tracked_ev"]["selected_station"] == "st-1"


def test_dashboard_uses_recommendation_history_and_ppo_aliases(tmp_path: Path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=1, station_count=1, tracked=1)

    state = {
        "simulation": {"step": 2, "time": 2.0, "running": True, "speed": 1.0, "fleet_size": 1, "ev_count": 1, "tracked_count": 1},
        "kpis": {"avg_speed": 5.0, "traffic_density": 0.2, "avg_wait_min": 0.3, "energy_consumed_kwh": 1.0},
        "vehicles": [{"id": "ev-1", "battery_pct": 60.0, "status": "driving", "current_position": {"x": 1.0, "y": 2.0}}],
        "station_details": [{"station_id": "st-1", "name": "North Station", "lat": 12.97, "lon": 77.59, "total_ports": 4, "available_ports": 2, "occupied_ports": 2, "queue_length": 1, "price_per_kwh": 0.22, "avg_wait_estimate": 3.0, "grid_load_kw": 40.0, "utilization": 0.5}],
        "traffic": [{"edge_id": "edge_1", "density": 1, "congestion": 0.1, "speed": 5.0}],
        "ppo": [{"ev_id": "ev-1", "action": "charge", "station_id": "st-1", "reward": 3.5, "battery": 60.0, "decision_time": 2, "decision_features": [0.2, 0.3]}],
        "tracked_ids": ["ev-1"],
        "metrics": [],
        "recommendations": [{"vehicle_id": "ev-1", "ppo_reward": 2.0, "step": 1}, {"vehicle_id": "ev-1", "ppo_reward": 3.5, "step": 2}],
        "charging_events": [],
        "latest_recommendation": None,
        "network": {},
        "xai": {},
    }

    dashboard.update(state)

    payload = json.loads((tmp_path / "dashboard_state.json").read_text(encoding="utf-8"))
    assert payload["history"]["rewards"] == [2.0, 3.5]
    assert payload["history"]["timestamps"] == [1, 2]
    assert payload["ppo_decision"]["ev_id"] == "ev-1"
    assert payload["ppo_decision"]["selected_station"] == "st-1"


def test_dashboard_html_uses_reason_from_latest_ppo(tmp_path: Path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=1, station_count=1, tracked=1)

    state = {
        "simulation": {"step": 2, "time": 2.0, "running": True, "speed": 1.0, "fleet_size": 1, "ev_count": 1, "tracked_count": 1},
        "kpis": {"avg_speed": 5.0, "traffic_density": 0.2, "avg_wait_min": 0.3, "energy_consumed_kwh": 1.0},
        "vehicles": [{"id": "ev-1", "battery_pct": 60.0, "status": "driving", "current_position": {"x": 1.0, "y": 2.0}}],
        "station_details": [{"station_id": "st-1", "name": "North Station", "lat": 12.97, "lon": 77.59, "total_ports": 4, "available_ports": 2, "occupied_ports": 2, "queue_length": 1, "price_per_kwh": 0.22, "avg_wait_estimate": 3.0, "grid_load_kw": 40.0, "utilization": 0.5}],
        "traffic": [{"edge_id": "edge_1", "density": 1, "congestion": 0.1, "speed": 5.0}],
        "ppo": [{"ev_id": "ev-1", "action": "charge", "station_id": "st-1", "reward": 3.5, "battery": 60.0, "reason": "lowest_wait"}],
        "tracked_ids": ["ev-1"],
        "metrics": [],
        "recommendations": [],
        "charging_events": [],
        "latest_recommendation": None,
        "network": {},
        "xai": {},
    }

    dashboard.update(state)

    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert "lowest_wait" in html


def test_dashboard_emits_live_telemetry_metadata(tmp_path: Path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=1, station_count=1, tracked=1)

    state = {
        "simulation": {"step": 4, "time": 4.0, "running": True, "speed": 1.0, "fleet_size": 1, "ev_count": 1, "tracked_count": 1},
        "kpis": {"avg_speed": 5.0, "traffic_density": 0.2, "avg_wait_min": 0.3, "energy_consumed_kwh": 1.0},
        "vehicles": [{"id": "ev-1", "battery_pct": 60.0, "status": "driving", "current_position": {"x": 1.0, "y": 2.0}}],
        "station_details": [{"station_id": "st-1", "name": "North Station", "lat": 12.97, "lon": 77.59, "total_ports": 4, "available_ports": 2, "occupied_ports": 2, "queue_length": 1, "price_per_kwh": 0.22, "avg_wait_estimate": 3.0, "grid_load_kw": 40.0, "utilization": 0.5}],
        "traffic": [{"edge_id": "edge_1", "density": 1, "congestion": 0.1, "speed": 5.0}],
        "ppo": [],
        "tracked_ids": ["ev-1"],
        "metrics": [],
        "recommendations": [],
        "charging_events": [],
        "latest_recommendation": None,
        "network": {},
        "xai": {},
    }

    dashboard.update(state)

    payload = json.loads((tmp_path / "dashboard_state.json").read_text(encoding="utf-8"))
    assert payload["telemetry"]["step"] == 4
    assert payload["telemetry"]["status"] in {"LIVE", "STALE", "DISCONNECTED"}
    assert payload["telemetry"]["age_ms"] >= 0


def test_dashboard_html_scales_coordinates_into_visible_network_view(tmp_path: Path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=1, station_count=2, tracked=1)

    state = {
        "simulation": {"step": 5, "time": 5.0, "running": True, "speed": 1.0, "fleet_size": 1, "ev_count": 1, "tracked_count": 1},
        "kpis": {"avg_speed": 6.0, "traffic_density": 0.3, "avg_wait_min": 0.5, "energy_consumed_kwh": 1.2},
        "vehicles": [{"id": "ev-1", "battery_pct": 55.0, "status": "driving", "current_position": {"x": 1250.0, "y": 3000.0}}],
        "station_details": [
            {"station_id": "st-1", "name": "North Station", "x": 1200.0, "y": 2800.0, "total_ports": 4, "available_ports": 2, "occupied_ports": 2, "queue_length": 1, "price_per_kwh": 0.22, "avg_wait_estimate": 3.0, "grid_load_kw": 40.0, "utilization": 0.5},
            {"station_id": "st-2", "name": "South Station", "x": 2500.0, "y": 3200.0, "total_ports": 4, "available_ports": 2, "occupied_ports": 2, "queue_length": 1, "price_per_kwh": 0.22, "avg_wait_estimate": 3.0, "grid_load_kw": 40.0, "utilization": 0.5},
        ],
        "traffic": [{"edge_id": "edge_1", "density": 1, "congestion": 0.1, "speed": 5.0}],
        "ppo": [],
        "tracked_ids": ["ev-1"],
        "metrics": [],
        "recommendations": [],
        "charging_events": [],
        "latest_recommendation": None,
        "network": {},
        "xai": {},
    }

    dashboard.update(state)

    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert "normalizeCoordinate" in html
    assert "xRange" in html


def test_dashboard_marks_telemetry_live_when_timestamp_is_missing(tmp_path: Path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=1, station_count=1, tracked=1)

    state = {
        "simulation": {"step": 6, "time": 6.0, "running": True, "speed": 1.0, "fleet_size": 1, "ev_count": 1, "tracked_count": 1},
        "kpis": {"avg_speed": 5.0, "traffic_density": 0.2, "avg_wait_min": 0.3, "energy_consumed_kwh": 1.0},
        "vehicles": [{"id": "ev-1", "battery_pct": 60.0, "status": "driving", "current_position": {"x": 1.0, "y": 2.0}}],
        "station_details": [{"station_id": "st-1", "name": "North Station", "lat": 12.97, "lon": 77.59, "total_ports": 4, "available_ports": 2, "occupied_ports": 2, "queue_length": 1, "price_per_kwh": 0.22, "avg_wait_estimate": 3.0, "grid_load_kw": 40.0, "utilization": 0.5}],
        "traffic": [{"edge_id": "edge_1", "density": 1, "congestion": 0.1, "speed": 5.0}],
        "ppo": [],
        "tracked_ids": ["ev-1"],
        "metrics": [],
        "recommendations": [],
        "charging_events": [],
        "latest_recommendation": None,
        "network": {},
        "xai": {},
    }

    dashboard.update(state)

    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    payload = json.loads((tmp_path / "dashboard_state.json").read_text(encoding="utf-8"))
    assert payload["telemetry"]["status"] == "LIVE"
    assert payload["telemetry"]["age_ms"] < 5000
    assert "yRange" in html
