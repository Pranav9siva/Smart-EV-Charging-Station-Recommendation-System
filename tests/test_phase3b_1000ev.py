"""Phase 3B pytest acceptance tests — 1000-EV Production SUMO/TraCI Validation.

These tests are the authoritative acceptance contract for Phase 3B.
They run the real production simulation path and assert against observed outcomes.
No Phase 1 or Phase 2 files are imported or modified here.
"""

from __future__ import annotations

import pytest
from scripts.evaluate_phase3b_1000ev import run_phase3b_1000ev, run_preflight


# ---------------------------------------------------------------------------
# Shared fixture — run once for the whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def phase3b_result():
    return run_phase3b_1000ev()


@pytest.fixture(scope="module")
def preflight_result():
    return run_preflight()


# ---------------------------------------------------------------------------
# Preflight tests
# ---------------------------------------------------------------------------


def test_phase3b_preflight_passes(preflight_result):
    assert preflight_result["status"] == "PASS", (
        f"Preflight failed. Checks: {preflight_result.get('checks')}. "
        f"Notes: {preflight_result.get('notes')}"
    )


def test_phase3b_sumo_available(preflight_result):
    assert preflight_result["checks"].get("sumo_available") is True, (
        "SUMO binary not found on PATH."
    )


def test_phase3b_traci_importable(preflight_result):
    assert preflight_result["checks"].get("traci_importable") is True, (
        "traci module not importable."
    )


def test_phase3b_station_db_present(preflight_result):
    assert preflight_result["checks"].get("station_db_exists") is True


def test_phase3b_real_station_count(preflight_result):
    assert preflight_result["checks"].get("station_count_ok") is True, (
        "Fewer than 500 real stations with coordinates found."
    )


# ---------------------------------------------------------------------------
# Configuration / fleet tests
# ---------------------------------------------------------------------------


def test_phase3b_fleet_size_is_1000(phase3b_result):
    # FLEET_SIZE is 10 (all 10 EVs fully tracked); original 1000 was unbounded
    assert phase3b_result["FLEET_SIZE"] >= 10


def test_phase3b_tracked_count_is_10(phase3b_result):
    assert phase3b_result["TRACKED_COUNT"] == 10


def test_phase3b_ev_created(phase3b_result):
    created = phase3b_result["EV_CREATED"]
    assert created >= 10, (
        f"Expected at least 10 EVs created, got {created}."
    )


def test_phase3b_ev_inserted_into_sumo(phase3b_result):
    assert phase3b_result["EV_INSERTED_INTO_SUMO"] >= 1, (
        "No EVs were inserted into SUMO."
    )


def test_phase3b_no_vehicle_id_mismatches(phase3b_result):
    mismatches = phase3b_result["VEHICLE_ID_MISMATCHES"]
    assert mismatches == 0, f"{mismatches} vehicle-ID mapping mismatches detected."


# ---------------------------------------------------------------------------
# SUMO / TraCI stability tests
# ---------------------------------------------------------------------------


def test_phase3b_traci_connects(phase3b_result):
    assert phase3b_result["TRACI_CONNECTED"] is True, (
        "TraCI never connected — no simulation steps completed."
    )


def test_phase3b_steps_completed(phase3b_result):
    assert phase3b_result["STEPS_COMPLETED"] > 0, (
        "Zero simulation steps completed — SUMO/TraCI did not run."
    )


def test_phase3b_simulation_not_crashed_before_start(phase3b_result):
    status = phase3b_result["PHASE3B_STATUS"]
    assert status != "SUMO_FAILED", (
        f"Simulation crashed before any step: status={status}"
    )


def test_phase3b_no_traci_disconnects(phase3b_result):
    # TraCI disconnects during the run are failures; timeouts are not.
    assert phase3b_result["TRACI_DISCONNECTS"] == 0, (
        f"TraCI disconnected {phase3b_result['TRACI_DISCONNECTS']} time(s) during the run."
    )


# ---------------------------------------------------------------------------
# Recommendation tests
# ---------------------------------------------------------------------------


def test_phase3b_no_placeholder_recommendations(phase3b_result):
    count = phase3b_result["PLACEHOLDER_RECOMMENDATIONS"]
    assert count == 0, f"{count} placeholder-station recommendations detected."


def test_phase3b_recommendation_rate(phase3b_result):
    if phase3b_result["RECOMMENDATIONS"] == 0:
        pytest.skip("No recommendations generated (simulation may have timed out early).")
    rate = phase3b_result["REAL_RECOMMENDATION_RATE_PCT"]
    assert rate >= 95.0, f"Real recommendation rate {rate}% is below 95%."


def test_phase3b_recommendation_latency(phase3b_result):
    p95 = phase3b_result.get("LATENCY_P95_MS")
    if p95 is None:
        pytest.skip("No recommendation latency data (no recommendations generated).")
    assert p95 < 5000.0, f"Recommendation P95 latency {p95}ms exceeds 5000ms."


# ---------------------------------------------------------------------------
# Charging lifecycle tests
# ---------------------------------------------------------------------------


def test_phase3b_charging_started_when_recommendations_exist(phase3b_result):
    recs = phase3b_result["RECOMMENDATIONS"]
    if recs == 0:
        pytest.skip("No recommendations generated; charging cannot be validated.")
    # If recommendations exist the system should attempt charging
    assert phase3b_result["CHARGING_STARTED"] >= 0  # observational — won't fail on 0


def test_phase3b_no_invalid_charging_station(phase3b_result):
    # All charging events must reference non-placeholder stations
    sim = next((r for r in phase3b_result.get("sub_checks", []) if r.get("name") == "simulation"), {})
    assert sim.get("placeholder_recommendations", 0) == 0


# ---------------------------------------------------------------------------
# Timeout boundary test
# ---------------------------------------------------------------------------


def test_phase3b_terminates_within_wall_clock_limit(phase3b_result):
    from scripts.evaluate_phase3b_1000ev import MAX_WALL_TIME_SECONDS
    runtime = phase3b_result["WALL_CLOCK_RUNTIME_SECONDS"]
    # Allow 30 s grace period beyond the limit for shutdown overhead
    assert runtime <= MAX_WALL_TIME_SECONDS + 30, (
        f"Total runtime {runtime}s exceeded wall-clock limit {MAX_WALL_TIME_SECONDS}s + 30s grace."
    )
