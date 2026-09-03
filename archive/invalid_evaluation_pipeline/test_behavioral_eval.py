from scripts.behavioral_eval_ppo import BehavioralEvaluationEnv


def test_behavioral_env_records_charging_when_vehicle_needs_charge() -> None:
    env = BehavioralEvaluationEnv(db_path="data/stations/stations.sqlite", tracked_vehicle_count=1, candidate_count=1)
    try:
        env.reset(seed=7)
        station = env.station_manager.list_all_stations()[0]
        station_id = station["station_id"]
        vehicle = env.vehicle_manager.list_tracked_vehicles()[0]
        vehicle.battery_pct = 10.0
        vehicle.remaining_range_km = 1.0
        env.current_candidates = [
            {
                "station_id": station_id,
                "distance_km": 2.0,
                "travel_time_min": 5.0,
                "free_ports": 1,
                "total_ports": 2,
                "price_per_kwh": 0.2,
                "avg_wait_min": 0.0,
                "queue_len": 0,
                "grid_load_kw": 10.0,
            }
        ]

        _, _, _, _, info = env.step(0)
        behavior = info["behavior"]

        assert behavior["charge_needed"] is True
        assert behavior["charging_event"] is True
        assert behavior["successful_charging_decision"] is True
        assert behavior["soc_after"] > behavior["soc_before"]
    finally:
        env.close()
