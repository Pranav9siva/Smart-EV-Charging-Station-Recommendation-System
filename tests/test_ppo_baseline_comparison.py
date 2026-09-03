from scripts.compare_ppo_baselines import _baseline_recommendation


class FakeStationManager:
    def list_all_stations(self):
        return [
            {"station_id": "near", "lat": 12.9716, "lon": 77.5946},
            {"station_id": "cheap", "lat": 12.9816, "lon": 77.6046},
            {"station_id": "wait", "lat": 12.9916, "lon": 77.6146},
        ]

    def get_station_metrics(self, station_id):
        values = {
            "near": {"available_ports": 1, "total_ports": 2, "price_per_kwh": 20.0, "avg_wait_estimate": 8.0, "queue_length": 2},
            "cheap": {"available_ports": 1, "total_ports": 2, "price_per_kwh": 5.0, "avg_wait_estimate": 5.0, "queue_length": 1},
            "wait": {"available_ports": 2, "total_ports": 2, "price_per_kwh": 12.0, "avg_wait_estimate": 0.0, "queue_length": 0},
        }
        return values[station_id]


def test_baseline_policies_select_their_defined_objective() -> None:
    manager = FakeStationManager()
    vehicle = {"lat": 12.9716, "lon": 77.5946, "battery_pct": 15.0, "battery_capacity_kwh": 60.0, "remaining_range_km": 50.0}

    assert _baseline_recommendation(manager, vehicle, "nearest")["station"] == "near"
    assert _baseline_recommendation(manager, vehicle, "cheapest")["station"] == "cheap"
    assert _baseline_recommendation(manager, vehicle, "minimum_wait")["station"] == "wait"
