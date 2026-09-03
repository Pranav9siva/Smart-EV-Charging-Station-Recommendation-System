from scripts.compare_ppo_checkpoints import is_placeholder_candidate, summarize_episode_metrics


def test_placeholder_candidate_detection() -> None:
    placeholder = {
        "station_id": "",
        "distance_km": 1e6,
        "travel_time_min": 1e6,
        "free_ports": 0,
        "total_ports": 0,
        "price_per_kwh": 0.0,
        "avg_wait_min": 1e6,
        "queue_len": 0,
        "grid_load_kw": 0.0,
    }
    real_candidate = {
        "station_id": "station_1",
        "distance_km": 2.1,
        "travel_time_min": 5.0,
        "free_ports": 2,
        "total_ports": 4,
        "price_per_kwh": 0.18,
        "avg_wait_min": 1.2,
        "queue_len": 1,
        "grid_load_kw": 50.0,
    }

    assert is_placeholder_candidate(placeholder)
    assert not is_placeholder_candidate(real_candidate)


def test_episode_summary_aggregates_metrics() -> None:
    step_rows = [
        {"reward": 1.0, "charging_event": True, "successful_charging_decision": True, "invalid_action": False, "completed": False},
        {"reward": 3.0, "charging_event": False, "successful_charging_decision": False, "invalid_action": True, "completed": True},
    ]

    summary = summarize_episode_metrics(step_rows, episode=1, seed=42)

    assert summary["reward_total"] == 4.0
    assert summary["reward_mean"] == 2.0
    assert summary["charging_events"] == 1
    assert summary["successful_charging_decisions"] == 1
    assert summary["invalid_actions"] == 1
    assert summary["completed"] is True
