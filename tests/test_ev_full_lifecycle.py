from __future__ import annotations

import multiprocessing as mp
import queue
import sqlite3
import time
import traceback
from pathlib import Path
from typing import Any

import pytest

TIMEOUT_SECONDS = 60


def _seed_phase1c_station_db(db_path: Path) -> None:
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stations (
                station_id TEXT PRIMARY KEY,
                name TEXT,
                lat REAL,
                lon REAL,
                operator TEXT,
                address TEXT,
                grid_zone_id TEXT,
                data_source TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ports (
                port_id TEXT PRIMARY KEY,
                station_id TEXT,
                connector_type TEXT,
                power_kw REAL,
                status TEXT,
                status_updated_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT,
                price_per_kwh REAL,
                tariff_period TEXT,
                recorded_at TEXT
            )
            """
        )

        now = "2026-01-01T00:00:00Z"
        stations = [
            ("phase1c_station_1", "Phase1C Station 1", 12.9716, 77.5946, "test", "A", "zone_1", "phase1c"),
            ("phase1c_station_2", "Phase1C Station 2", 12.9720, 77.5950, "test", "B", "zone_1", "phase1c"),
            ("phase1c_station_3", "Phase1C Station 3", 12.9724, 77.5954, "test", "C", "zone_1", "phase1c"),
        ]
        connection.executemany(
            "INSERT OR REPLACE INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            stations,
        )

        ports = [
            ("phase1c_station_1_p1", "phase1c_station_1", "CCS", 1000.0, "Available", now),
            ("phase1c_station_2_p1", "phase1c_station_2", "CCS", 350.0, "Available", now),
            ("phase1c_station_3_p1", "phase1c_station_3", "CCS", 120.0, "Available", now),
        ]
        connection.executemany(
            "INSERT OR REPLACE INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ports,
        )

        prices = [
            ("phase1c_station_1", 0.20, "off_peak", now),
            ("phase1c_station_2", 0.24, "off_peak", now),
            ("phase1c_station_3", 0.28, "off_peak", now),
        ]
        connection.executemany(
            "INSERT INTO price_history (station_id, price_per_kwh, tariff_period, recorded_at) VALUES (?, ?, ?, ?)",
            prices,
        )

        connection.commit()
    finally:
        connection.close()


def _fail_payload(message: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {"ok": False, "error": message, "diagnostics": diagnostics}


def _phase1c_worker(repo_root: str, db_path: str, result_queue: mp.Queue) -> None:
    controller = None
    traci_module = None
    last_successful_step = -1
    diagnostics: dict[str, Any] = {
        "EV_ID": "",
        "current_SUMO_edge": "",
        "target_station_ID": "",
        "EV_SOC": None,
        "remaining_range": None,
        "station_distance": None,
        "station_free_ports": None,
        "station_queue_length": None,
        "charging_state": "",
        "current_simulation_step": -1,
        "TraCI_connection_state": False,
        "last_successful_SUMO_step": -1,
    }

    try:
        from src.simulation.controller import SimulationController

        _seed_phase1c_station_db(Path(db_path))

        sumo_cfg = Path(repo_root) / "simulations" / "bangalore" / "sim.sumocfg"
        model_path = Path(repo_root) / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"

        controller = SimulationController(
            sumo_cfg=str(sumo_cfg),
            db_path=str(db_path),
            model_path=str(model_path),
            fleet_size=1,
            station_count=3,
            tracked=1,
            charge_threshold_pct=20.0,
            use_gui=False,
            simulation_seed=123,
        )

        controller.start()
        traci_module = __import__("traci")

        connected = bool(controller._is_traci_connected())
        diagnostics["TraCI_connection_state"] = connected
        if not connected:
            result_queue.put(_fail_payload("SUMO/TraCI connection failed.", diagnostics))
            return

        vehicles = controller.vehicle_manager.list_vehicles()
        if len(vehicles) != 1:
            result_queue.put(_fail_payload(f"Expected exactly 1 managed EV, found {len(vehicles)}.", diagnostics))
            return

        vehicle = vehicles[0]
        ev_id = str(vehicle.vehicle_id)
        diagnostics["EV_ID"] = ev_id

        sumo_id = controller.vm_to_sumo.get(ev_id, ev_id)
        sumo_ids = set(traci_module.vehicle.getIDList())
        if sumo_id not in sumo_ids:
            result_queue.put(_fail_payload(f"EV insertion/mapping failed for {ev_id}.", diagnostics))
            return

        route_edges = list(traci_module.vehicle.getRoute(sumo_id))
        if not route_edges:
            result_queue.put(_fail_payload(f"EV {ev_id} has no valid SUMO route.", diagnostics))
            return

        # Move station coordinates near the current EV position to make reroute deterministic.
        x, y = traci_module.vehicle.getPosition(sumo_id)
        lon, lat = traci_module.simulation.convertGeo(x, y)
        conn = controller.station_manager.repo.connection
        conn.execute("UPDATE stations SET lat = ?, lon = ? WHERE station_id = ?", (float(lat), float(lon), "phase1c_station_1"))
        conn.execute("UPDATE stations SET lat = ?, lon = ? WHERE station_id = ?", (float(lat) + 0.0003, float(lon) + 0.0003, "phase1c_station_2"))
        conn.execute("UPDATE stations SET lat = ?, lon = ? WHERE station_id = ?", (float(lat) + 0.0006, float(lon) + 0.0006, "phase1c_station_3"))
        conn.commit()
        controller._invalidate_station_snapshots()

        # Deterministic low SOC trigger.
        vehicle.battery_pct = 12.0
        vehicle.remaining_range_km = 1.0
        diagnostics["EV_SOC"] = float(vehicle.battery_pct)
        diagnostics["remaining_range"] = float(vehicle.remaining_range_km)

        lifecycle_trace = [
            "EV created",
            "SUMO connected",
            "EV moving",
            f"SOC = {vehicle.battery_pct:.1f}%",
        ]

        charging_required = False
        real_station_selected = False
        charging_started = False
        charging_completed = False
        port_released = False
        ev_resumed = False
        selected_station_id = ""
        soc_before = float(vehicle.battery_pct)
        soc_after = float(vehicle.battery_pct)
        range_before = float(vehicle.remaining_range_km)
        range_after = float(vehicle.remaining_range_km)

        max_steps = 1200
        for step in range(max_steps):
            diagnostics["current_simulation_step"] = int(step)
            try:
                traci_module.simulationStep()
                last_successful_step = step
                diagnostics["last_successful_SUMO_step"] = int(last_successful_step)
                diagnostics["TraCI_connection_state"] = bool(controller._is_traci_connected())
            except Exception as exc:
                result_queue.put(_fail_payload(f"TraCI/SUMO step failure: {exc}", diagnostics))
                return

            controller._update_vehicles(step)
            controller._process_charging(step)
            controller._invalidate_station_snapshots()

            try:
                diagnostics["current_SUMO_edge"] = str(traci_module.vehicle.getRoadID(sumo_id))
            except Exception:
                diagnostics["current_SUMO_edge"] = ""

            diagnostics["EV_SOC"] = float(vehicle.battery_pct)
            diagnostics["remaining_range"] = float(vehicle.remaining_range_km)

            if vehicle.battery_pct <= controller.charge_threshold_pct and ev_id not in controller.assignments:
                charging_required = True

            if charging_required and "charging required" not in " ".join(lifecycle_trace):
                lifecycle_trace.append("charging required")

            snapshots = controller._get_station_snapshots(force_refresh=True)
            candidate_ids = [str(item.get("station_id") or "") for item in snapshots if str(item.get("station_id") or "")]
            if charging_required and not candidate_ids:
                result_queue.put(_fail_payload("Candidate station list is empty.", diagnostics))
                return

            if controller.recommendation_log:
                latest_rec = controller.recommendation_log[-1]
                selected_station_id = str(latest_rec.get("selected_station") or "")
                diagnostics["target_station_ID"] = selected_station_id
                diagnostics["station_distance"] = latest_rec.get("distance_km")
                diagnostics["station_free_ports"] = latest_rec.get("available_ports")
                diagnostics["station_queue_length"] = latest_rec.get("queue_length")

                if selected_station_id and selected_station_id != "_empty_candidate":
                    real_station_selected = True

                if selected_station_id:
                    station_metrics = controller.station_manager.get_station_metrics(selected_station_id) or {}
                    free_ports = int(station_metrics.get("available_ports", 0) or 0)
                    queue_len = int(station_metrics.get("queue_length", 0) or 0)
                    diagnostics["station_free_ports"] = free_ports
                    diagnostics["station_queue_length"] = queue_len

                    if free_ports < 0:
                        result_queue.put(_fail_payload("Invalid station free port data.", diagnostics))
                        return

            assignment = controller.assignments.get(ev_id)
            if assignment is not None:
                diagnostics["charging_state"] = str(assignment.status)

            charging_started_event = next((e for e in reversed(controller.charging_events) if e.get("vehicle_id") == ev_id and e.get("event_type") == "charging_started"), None)
            if charging_started_event is not None and not charging_started:
                charging_started = True
                selected_station_id = str(charging_started_event.get("station_id") or selected_station_id)
                diagnostics["target_station_ID"] = selected_station_id
                soc_before = float(charging_started_event.get("battery_pct") or vehicle.battery_pct)
                range_before = float(vehicle.remaining_range_km)
                lifecycle_trace.append(f"selected station = {selected_station_id}")
                lifecycle_trace.append(f"distance = {diagnostics['station_distance']} km")
                lifecycle_trace.append("EV reached station")
                lifecycle_trace.append("charging started")

                port_id = str(charging_started_event.get("port_id") or "")
                if port_id:
                    state = controller.station_manager.get_station_state(selected_station_id) or {}
                    ports = state.get("ports", [])
                    occupied = any(str(p.get("port_id")) == port_id and str(p.get("status")) == "Charging" for p in ports)
                    if not occupied:
                        result_queue.put(_fail_payload("Station port did not become occupied during charging.", diagnostics))
                        return

            charging_completed_event = next((e for e in reversed(controller.charging_events) if e.get("vehicle_id") == ev_id and e.get("event_type") == "charging_completed"), None)
            if charging_completed_event is not None and not charging_completed:
                charging_completed = True
                soc_after = float(charging_completed_event.get("battery_pct") or vehicle.battery_pct)
                range_after = float(vehicle.remaining_range_km)
                lifecycle_trace.append(f"SOC {soc_before:.2f}% -> {soc_after:.2f}%")

                port_id = str(charging_completed_event.get("port_id") or "")
                if port_id and selected_station_id:
                    state = controller.station_manager.get_station_state(selected_station_id) or {}
                    ports = state.get("ports", [])
                    released = any(str(p.get("port_id")) == port_id and str(p.get("status")) == "Available" for p in ports)
                    port_released = bool(released)

            resumed_event = next((e for e in reversed(controller.charging_events) if e.get("vehicle_id") == ev_id and e.get("event_type") == "resumed_route"), None)
            if resumed_event is not None and not ev_resumed:
                ev_resumed = True
                lifecycle_trace.append("port released")
                lifecycle_trace.append("EV resumed")

            if charging_required and real_station_selected and charging_started and charging_completed and port_released and ev_resumed:
                break

        diagnostics["last_successful_SUMO_step"] = int(last_successful_step)

        if not charging_required:
            result_queue.put(_fail_payload("Charging was never required for the EV.", diagnostics))
            return
        if not real_station_selected:
            result_queue.put(_fail_payload("No real station was selected (selected station empty or placeholder).", diagnostics))
            return
        if diagnostics.get("station_distance") is None:
            result_queue.put(_fail_payload("Selected station distance data is missing.", diagnostics))
            return
        try:
            station_distance = float(diagnostics.get("station_distance"))
        except Exception:
            result_queue.put(_fail_payload("Selected station distance data is not numeric.", diagnostics))
            return
        if station_distance >= 1_000_000.0 or station_distance < 0.0:
            result_queue.put(_fail_payload("Selected station distance data is placeholder/invalid.", diagnostics))
            return
        if diagnostics.get("station_free_ports") is None:
            result_queue.put(_fail_payload("Selected station free-port data is missing.", diagnostics))
            return
        if not charging_started:
            result_queue.put(_fail_payload("Charging never started.", diagnostics))
            return
        if not charging_completed:
            result_queue.put(_fail_payload("Charging never completed within bounded steps.", diagnostics))
            return
        if not port_released:
            result_queue.put(_fail_payload("Station port was not released after charging.", diagnostics))
            return
        if not ev_resumed:
            result_queue.put(_fail_payload("EV did not resume movement after charging.", diagnostics))
            return

        if not (soc_after > soc_before):
            result_queue.put(_fail_payload("SOC did not increase after charging.", diagnostics))
            return
        if not (range_after > range_before):
            result_queue.put(_fail_payload("Range did not increase after charging.", diagnostics))
            return

        payload = {
            "PHASE1C_STATUS": "PASS",
            "EV_CREATED": True,
            "SUMO_CONNECTED": True,
            "VALID_ROUTE": True,
            "CHARGING_REQUIRED": charging_required,
            "REAL_STATION_SELECTED": real_station_selected,
            "CHARGING_STARTED": charging_started,
            "SOC_BEFORE": round(float(soc_before), 4),
            "SOC_AFTER": round(float(soc_after), 4),
            "RANGE_BEFORE": round(float(range_before), 4),
            "RANGE_AFTER": round(float(range_after), 4),
            "PORT_RELEASED": port_released,
            "EV_RESUMED": ev_resumed,
            "TRACE": " -> ".join(lifecycle_trace),
            "DIAGNOSTICS": diagnostics,
        }
        result_queue.put({"ok": True, "payload": payload})
    except Exception:
        diagnostics["last_successful_SUMO_step"] = int(last_successful_step)
        result_queue.put(_fail_payload(traceback.format_exc(), diagnostics))
    finally:
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass


def _run_phase1c_with_timeout(repo_root: Path, db_path: Path, timeout_seconds: int = TIMEOUT_SECONDS) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_phase1c_worker, args=(str(repo_root), str(db_path), result_queue))

    started_at = time.perf_counter()
    process.start()
    process.join(timeout_seconds)
    elapsed_seconds = time.perf_counter() - started_at

    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail(f"Phase 1C exceeded hard timeout of {timeout_seconds} seconds.")

    if process.exitcode not in (0, None):
        pytest.fail(f"Phase 1C worker exited with code {process.exitcode}.")

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        pytest.fail("Phase 1C worker produced no result.")

    if not result.get("ok"):
        diagnostics = result.get("diagnostics", {})
        pytest.fail(
            "\n".join(
                [
                    f"Phase 1C failed: {result.get('error', 'unknown error')}",
                    f"EV ID: {diagnostics.get('EV_ID')}",
                    f"current SUMO edge: {diagnostics.get('current_SUMO_edge')}",
                    f"target station ID: {diagnostics.get('target_station_ID')}",
                    f"EV SOC: {diagnostics.get('EV_SOC')}",
                    f"remaining range: {diagnostics.get('remaining_range')}",
                    f"station distance: {diagnostics.get('station_distance')}",
                    f"station free ports: {diagnostics.get('station_free_ports')}",
                    f"station queue length: {diagnostics.get('station_queue_length')}",
                    f"charging state: {diagnostics.get('charging_state')}",
                    f"current simulation step: {diagnostics.get('current_simulation_step')}",
                    f"TraCI connection state: {diagnostics.get('TraCI_connection_state')}",
                    f"last successful SUMO step: {diagnostics.get('last_successful_SUMO_step')}",
                ]
            )
        )

    payload = dict(result["payload"])
    payload["ELAPSED_SECONDS"] = round(float(elapsed_seconds), 4)
    return payload


def test_phase1c_full_single_ev_charging_lifecycle(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "phase1c_stations.sqlite"

    report = _run_phase1c_with_timeout(repo_root=repo_root, db_path=db_path, timeout_seconds=TIMEOUT_SECONDS)

    print(report["TRACE"])
    print(f"PHASE1C_STATUS={report['PHASE1C_STATUS']}")
    print(f"ELAPSED_SECONDS={report['ELAPSED_SECONDS']}")
    print(f"EV_CREATED={report['EV_CREATED']}")
    print(f"SUMO_CONNECTED={report['SUMO_CONNECTED']}")
    print(f"VALID_ROUTE={report['VALID_ROUTE']}")
    print(f"CHARGING_REQUIRED={report['CHARGING_REQUIRED']}")
    print(f"REAL_STATION_SELECTED={report['REAL_STATION_SELECTED']}")
    print(f"CHARGING_STARTED={report['CHARGING_STARTED']}")
    print(f"SOC_BEFORE={report['SOC_BEFORE']}")
    print(f"SOC_AFTER={report['SOC_AFTER']}")
    print(f"RANGE_BEFORE={report['RANGE_BEFORE']}")
    print(f"RANGE_AFTER={report['RANGE_AFTER']}")
    print(f"PORT_RELEASED={report['PORT_RELEASED']}")
    print(f"EV_RESUMED={report['EV_RESUMED']}")

    assert report["PHASE1C_STATUS"] == "PASS"
    assert report["EV_CREATED"] is True
    assert report["SUMO_CONNECTED"] is True
    assert report["VALID_ROUTE"] is True
    assert report["CHARGING_REQUIRED"] is True
    assert report["REAL_STATION_SELECTED"] is True
    assert report["CHARGING_STARTED"] is True
    assert report["SOC_AFTER"] > report["SOC_BEFORE"]
    assert report["RANGE_AFTER"] > report["RANGE_BEFORE"]
    assert report["PORT_RELEASED"] is True
    assert report["EV_RESUMED"] is True
