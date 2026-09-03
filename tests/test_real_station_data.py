from __future__ import annotations

import time

from scripts.evaluate_phase2_recommendation import run_phase2a_real_station_data_validation


def test_phase2a_real_station_data_validation() -> None:
    started = time.perf_counter()
    result = run_phase2a_real_station_data_validation()
    elapsed = time.perf_counter() - started

    assert elapsed <= 30.0
    assert result["REAL_STATION_COUNT"] >= 500
    assert result["VALID_STATION_ID_COUNT"] == result["REAL_STATION_COUNT"]
    assert result["VALID_COORDINATE_COUNT"] == result["REAL_STATION_COUNT"]
    assert result["VALID_PORT_COUNT"] > 0
    assert result["VALID_PRICE_COUNT"] > 0
    assert result["VALID_QUEUE_COUNT"] > 0
    assert result["VALID_FREE_PORT_COUNT"] > 0
    assert result["PLACEHOLDER_STATION_COUNT"] == 0
