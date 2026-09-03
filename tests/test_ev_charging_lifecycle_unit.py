from __future__ import annotations

import multiprocessing as mp
import queue
import sqlite3
import traceback
from pathlib import Path
from typing import Any

import pytest

from src.ev_management.vehicle_manager import VehicleManager, VehicleRecord
from src.ev_model.battery import remaining_range_km
from src.station_management.manager import ChargingStationManager
from src.station_management.station_repository import StationRepository

TIMEOUT_SECONDS = 30


def _seed_temp_station_db(db_path: Path) -> None:
    connection = sqlite3.connect(str(db_path))
    try:
        repository = StationRepository(connection)
        now = "2026-01-01T00:00:00Z"

        connection.executemany(
            """
            INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("station_a", "Station A", 12.9716, 77.5946, "test", "A", "zone_a", "unit_test"),
                ("station_b", "Station B", 12.9720, 77.5950, "test", "B", "zone_b", "unit_test"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("station_a_port_1", "station_a", "CCS", 60.0, "Available", now),
                ("station_b_port_1", "station_b", "CCS", 50.0, "Available", now),
            ],
        )
        connection.executemany(
            """
            INSERT INTO price_history (station_id, price_per_kwh, tariff_period, recorded_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("station_a", 0.22, "off_peak", now),
                ("station_b", 0.30, "off_peak", now),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def _phase1a_lifecycle_worker(db_path: str, result_queue: mp.Queue) -> None:
    manager: ChargingStationManager | None = None
    try:
        db_file = Path(db_path)
        _seed_temp_station_db(db_file)

        manager = ChargingStationManager(db_path=str(db_file))

        vehicle = VehicleRecord(
            vehicle_id="ev_1",
            source="node_1",
            destination="node_2",
            battery_capacity_kwh=50.0,
            battery_pct=10.0,
            remaining_range_km=1.0,
            charging_requirement_kwh=45.0,
            tracked=True,
        )
        vehicle_manager = VehicleManager(vehicles=[vehicle], detail_count=1)

        tracked_vehicle = vehicle_manager.list_tracked_vehicles()[0]
        charge_needed = bool(tracked_vehicle.battery_pct <= 20.0 or tracked_vehicle.remaining_range_km <= 2.0)

        snapshots = manager.get_station_snapshots(force_refresh=True)
        distance_lookup = {"station_a": 2.0, "station_b": 4.0}

        candidates: list[dict[str, Any]] = []
        for snapshot in snapshots:
            station_id = str(snapshot["station_id"])
            candidates.append(
                {
                    "station_id": station_id,
                    "distance_km": float(distance_lookup[station_id]),
                    "price_per_kwh": float(snapshot.get("price_per_kwh") or 0.0),
                    "free_ports": int(snapshot.get("available_ports", 0) or 0),
                    "queue_len": int(snapshot.get("queue_length", 0) or 0),
                }
            )

        available_candidates = [candidate for candidate in candidates if candidate["free_ports"] > 0]
        selected_candidate = min(
            available_candidates,
            key=lambda item: (item["distance_km"], item["price_per_kwh"], item["queue_len"], item["station_id"]),
        )

        soc_before = float(tracked_vehicle.battery_pct)
        range_before = float(tracked_vehicle.remaining_range_km)

        reserved_port = manager.reserve_port(str(selected_candidate["station_id"]))
        charging_event = bool(charge_needed and reserved_port)
        successful_charging_decision = bool(charging_event)

        if successful_charging_decision:
            tracked_vehicle.battery_pct = 35.0
            tracked_vehicle.remaining_range_km = round(
                remaining_range_km(
                    soc_pct=tracked_vehicle.battery_pct,
                    battery_kwh=tracked_vehicle.battery_capacity_kwh,
                    consumption_wh_per_km=200.0,
                ),
                3,
            )
            manager.release_port(str(reserved_port))

        soc_after = float(tracked_vehicle.battery_pct)
        range_after = float(tracked_vehicle.remaining_range_km)

        station_state_after = manager.get_station_metrics(str(selected_candidate["station_id"])) or {}
        station_free_port_restored = bool(int(station_state_after.get("available_ports", 0) or 0) >= 1)

        payload = {
            "vehicle_id": tracked_vehicle.vehicle_id,
            "initial_soc": soc_before,
            "initial_range_km": range_before,
            "charge_needed": charge_needed,
            "selected_station_id": selected_candidate["station_id"],
            "charging_event": charging_event,
            "successful_charging_decision": successful_charging_decision,
            "final_soc": soc_after,
            "final_range_km": range_after,
            "station_free_port_restored": station_free_port_restored,
            "resumed_journey": bool(successful_charging_decision),
        }
        result_queue.put({"ok": True, "payload": payload})
    except Exception:
        result_queue.put({"ok": False, "error": traceback.format_exc()})
    finally:
        if manager is not None:
            manager.close()


def _run_lifecycle_with_timeout(db_path: Path, timeout_seconds: int = TIMEOUT_SECONDS) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_phase1a_lifecycle_worker, args=(str(db_path), result_queue))

    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail(f"Phase 1A lifecycle test exceeded {timeout_seconds} seconds timeout.")

    if process.exitcode not in (0, None):
        pytest.fail(f"Phase 1A lifecycle worker exited with code {process.exitcode}.")

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        pytest.fail("Phase 1A lifecycle worker produced no result.")

    if not result.get("ok"):
        pytest.fail(f"Phase 1A lifecycle worker failed:\n{result.get('error', 'unknown error')}")

    return dict(result["payload"])


def test_ev_charging_lifecycle_unit(tmp_path: Path) -> None:
    db_path = tmp_path / "phase1a_unit_stations.sqlite"
    result = _run_lifecycle_with_timeout(db_path=db_path, timeout_seconds=TIMEOUT_SECONDS)

    assert result["initial_soc"] == 10.0
    assert result["initial_range_km"] == 1.0
    assert result["charge_needed"] is True
    assert result["charging_event"] is True
    assert result["successful_charging_decision"] is True
    assert result["final_soc"] > result["initial_soc"]
    assert result["final_range_km"] > result["initial_range_km"]
    assert result["station_free_port_restored"] is True
    assert result["resumed_journey"] is True
