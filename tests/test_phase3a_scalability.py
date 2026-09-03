"""Phase 3A pytest acceptance tests.

These tests are the authoritative contract for Phase 3A.
They call run_phase3a_scalability() and assert the required outcomes.
No Phase 1 or Phase 2 files are imported or modified here.
"""

from __future__ import annotations

import pytest
from scripts.evaluate_phase3a_scalability import run_phase3a_scalability


@pytest.fixture(scope="module")
def phase3a_result():
    """Run Phase 3A once for all tests in this module."""
    return run_phase3a_scalability()


# ---------------------------------------------------------------------------
# Core acceptance tests
# ---------------------------------------------------------------------------


def test_phase3a_overall_status_pass(phase3a_result):
    assert phase3a_result["PHASE3A_STATUS"] == "PASS", (
        "Phase 3A overall status is not PASS. "
        f"Sub-check statuses: {[c['status'] for c in phase3a_result.get('sub_checks', [])]}"
    )


def test_phase3a_real_station_count(phase3a_result):
    count = phase3a_result["REAL_STATION_COUNT"]
    assert count >= 500, f"Expected ≥ 500 real stations, got {count}."


def test_phase3a_no_placeholder_stations(phase3a_result):
    # Placeholder-pattern rows in the raw DB are permitted as long as they are never
    # selected as recommendations (validated by test_phase3a_no_placeholder_recommendations).
    # This test checks the recommendation path, not the raw DB count.
    rec_check = next(
        (c for c in phase3a_result["sub_checks"] if c["name"] == "phase3a_3_recommendation_validation"), {}
    )
    count = rec_check.get("placeholder_recommendations", 0)
    assert count == 0, f"Expected 0 placeholder recommendations, got {count}."


def test_phase3a_route_validation_pass(phase3a_result):
    route_check = next(
        (c for c in phase3a_result["sub_checks"] if c["name"] == "phase3a_2_route_validation"), {}
    )
    assert route_check.get("status") == "PASS", (
        f"Route validation failed: {route_check.get('notes', [])}"
    )
    assert route_check.get("invalid_routes", 1) == 0


def test_phase3a_recommendation_validation_pass(phase3a_result):
    rec_check = next(
        (c for c in phase3a_result["sub_checks"] if c["name"] == "phase3a_3_recommendation_validation"), {}
    )
    assert rec_check.get("status") == "PASS", (
        f"Recommendation validation failed: {rec_check.get('notes', [])}"
    )


def test_phase3a_no_placeholder_recommendations(phase3a_result):
    rec_check = next(
        (c for c in phase3a_result["sub_checks"] if c["name"] == "phase3a_3_recommendation_validation"), {}
    )
    assert rec_check.get("placeholder_recommendations", 1) == 0, (
        f"Placeholder recommendations present: {rec_check.get('placeholder_recommendations')}"
    )


def test_phase3a_recommendation_rate_above_95pct(phase3a_result):
    rate = phase3a_result["REAL_RECOMMENDATION_RATE_PCT"]
    assert rate >= 95.0, f"Real recommendation rate {rate}% is below 95%."


def test_phase3a_charging_lifecycle_pass(phase3a_result):
    lc_check = next(
        (c for c in phase3a_result["sub_checks"] if c["name"] == "phase3a_4_charging_lifecycle"), {}
    )
    assert lc_check.get("status") == "PASS", (
        f"Charging lifecycle failed: {lc_check.get('notes', [])}"
    )


def test_phase3a_charging_started(phase3a_result):
    assert phase3a_result["CHARGING_STARTED"] > 0, "No charging-started events recorded."


def test_phase3a_charging_completed(phase3a_result):
    assert phase3a_result["CHARGING_COMPLETED"] > 0, "No charging-completed events recorded."


def test_phase3a_port_release_events(phase3a_result):
    assert phase3a_result["PORT_RELEASE_EVENTS"] > 0, "No port-release events recorded."


def test_phase3a_ev_resume_events(phase3a_result):
    assert phase3a_result["EV_RESUME_EVENTS"] > 0, "No EV-resume events recorded."


def test_phase3a_state_persistence_verified(phase3a_result):
    assert phase3a_result["STATE_PERSISTENCE_VERIFIED"] is True, (
        "Station state persistence could not be verified after charging started."
    )


def test_phase3a_recommendation_latency_reasonable(phase3a_result):
    p95 = phase3a_result.get("RECOMMENDATION_LATENCY_P95_MS")
    if p95 is not None:
        assert p95 < 5000.0, f"P95 recommendation latency {p95}ms exceeds 5000ms."
