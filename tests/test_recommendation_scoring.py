from __future__ import annotations

import time

from scripts.evaluate_phase2_recommendation import run_phase2c_recommendation_scoring_validation


def test_phase2c_recommendation_scoring_validation() -> None:
    started = time.perf_counter()
    result = run_phase2c_recommendation_scoring_validation()
    elapsed = time.perf_counter() - started

    assert elapsed <= 30.0

    details = result["details"]
    assert details["scenario_a_distance_prefers_closer"] is True
    assert details["scenario_b_cost_prefers_cheaper"] is True
    assert details["scenario_c_ports_prefers_more_free_ports"] is True
    assert details["scenario_d_wait_prefers_lower_wait"] is True
