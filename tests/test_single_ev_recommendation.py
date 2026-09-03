from __future__ import annotations

import time

from scripts.evaluate_phase2_recommendation import run_phase2b_single_ev_recommendation


def test_phase2b_single_ev_recommendation() -> None:
    started = time.perf_counter()
    result = run_phase2b_single_ev_recommendation()
    elapsed = time.perf_counter() - started

    assert elapsed <= 30.0
    assert result["selected_station_is_real"] is True
    assert result["selected_station_metrics_available"] is True
    assert result["candidate_count"] > 0
    assert result["placeholder_candidate_count"] == 0
    assert result["invalid_candidate_count"] == 0

    for row in result["candidates"]:
        assert row["station_id"]
        assert row["distance_km"] < 1_000_000.0
        assert row["travel_time_min"] < 1_000_000.0
        assert row["total_ports"] >= 1
        assert row["free_ports"] >= 0
