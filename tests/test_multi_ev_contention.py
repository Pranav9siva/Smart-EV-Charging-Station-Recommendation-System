from __future__ import annotations

import time

from scripts.evaluate_phase2_recommendation import run_phase2d_multi_ev_contention


def test_phase2d_multi_ev_contention() -> None:
    started = time.perf_counter()
    result = run_phase2d_multi_ev_contention()
    elapsed = time.perf_counter() - started

    assert elapsed <= 60.0
    assert result["status"] == "PASS"

    details = result["details"]
    checks = result["checks"]

    assert details["recommendation_count"] > 0
    assert details["real_recommendation_count"] > 0
    assert details["placeholder_recommendation_count"] == 0
    assert details["charging_started_count"] > 0
    assert details["charging_completed_count"] >= 0

    assert checks["different_evs_receive_recommendations"] is True
    assert checks["multiple_evs_can_select_same_station"] is True
    assert checks["charging_events_occur"] is True
    assert checks["station_state_persistent"] is True
    assert checks["ev_state_persistent"] is True
    assert checks["no_placeholder_selection"] is True
