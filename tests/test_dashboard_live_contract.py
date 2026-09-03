import json
from pathlib import Path

from src.dashboard.dashboard import Dashboard


def test_dashboard_canonical_state_exposes_live_contract_fields(tmp_path: Path) -> None:
    dashboard = Dashboard(output_dir=str(tmp_path), fleet_size=2, station_count=1, tracked=1)

    state = {
        "simulation": {"step": 7, "time": 7.5, "running": True, "speed": 1.5, "fleet_size": 2, "ev_count": 2, "tracked_count": 1, "simulation_status": "RUNNING"},
        "kpis": {"avg_speed": 18.0, "traffic_density": 0.75, "avg_wait_min": 3.5, "energy_consumed_kwh": 14.2},
        "vehicles": [
            {
                "id": "ev-1",
                "vehicle_id": "ev-1",
                "battery_pct": 62.0,
                "speed": 12.4,
                "current_position": {"x": 100.0, "y": 220.0},
                "status": "driving",
                "current_edge": "edge_1",
                "destination": "edge_2",
                "recommended_station": "st-1",
                "ppo_reward": 2.3,
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
        "traffic": [{"edge_id": "edge_1", "density": 2, "congestion": 0.25, "speed": 12.4}],
        "ppo": [{"vehicle_id": "ev-1", "action": "charge", "station_id": "st-1", "reward": 2.3, "battery": 62.0, "reason": "lowest_wait"}],
        "tracked_ids": ["ev-1"],
        "metrics": [{"step": 7, "total_vehicles": 2, "assigned": 1, "charging": 0, "waiting": 1, "avg_wait_min": 3.0}],
        "recommendations": [{"vehicle_id": "ev-1", "ppo_reward": 2.3, "step": 7}],
        "charging_events": [],
        "latest_recommendation": {"vehicle_id": "ev-1", "selected_station": "st-1", "recommendation_reason": "lowest_wait"},
        "network": {},
        "xai": {},
    }

    dashboard.update(state)

    payload = json.loads((tmp_path / "dashboard_state.json").read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["network"]["stations"][0]["station_id"] == "st-1"
    assert payload["network"]["vehicles"][0]["vehicle_id"] == "ev-1"
    assert payload["tracked_ev"]["soc"] == 0.62
    assert payload["ppo_decision"]["reason"] == "lowest_wait"
    assert payload["stations"][0]["utilization"] == 0.5
    assert payload["stations"][0]["queue_length"] == 1
    assert payload["history"]["rewards"] == [2.3]
    assert payload["history"]["timestamps"] == [7]
