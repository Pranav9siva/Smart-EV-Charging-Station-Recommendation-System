import sqlite3
from datetime import timedelta
from pathlib import Path

from src.station_management.occupancy_simulator import OccupancySimulator, PortState, STATUS_TRANSITIONS
from src.station_management.pricing import PricingManager, current_tariff_period
from src.station_management.manager import ChargingStationManager
from src.station_management.station_repository import StationRepository
from scripts.seed_stations import create_db


def test_station_repository_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "stations.sqlite"
    conn = create_db(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("test_s1", "Test Station", 12.95, 77.60, "Operator", "Bangalore", "zone1", "simulated"),
    )
    cursor.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?) ",
        ("test_s1_p1", "test_s1", "Type2", 22.0, "Available", "2026-01-01T00:00:00"),
    )
    conn.commit()
    repo = StationRepository(conn)

    nearby = repo.get_nearby_stations(12.95, 77.60, 5.0)
    assert len(nearby) == 1
    state = repo.get_station_state("test_s1")
    assert state is not None
    assert state["station_id"] == "test_s1"
    assert state["price_per_kwh"] is None

    repo.log_status_change("test_s1_p1", "Charging")
    history = repo.get_occupancy_history("test_s1_p1", timedelta(days=1))
    assert len(history) == 1
    assert history[0]["status"] == "Charging"

    aggregate = repo.get_station_aggregate("test_s1")
    assert aggregate["available_ports"] == 0
    assert aggregate["occupied_ports"] == 1
    conn.close()


def test_pricing_manager_records_price(tmp_path: Path) -> None:
    db_path = tmp_path / "stations.sqlite"
    conn = create_db(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("test_s1", "Test Station", 12.95, 77.60, "Operator", "Bangalore", "zone1", "simulated"),
    )
    conn.commit()
    pricing = PricingManager(db_path)
    pricing.record_price("test_s1", 5.5, current_tariff_period())
    assert pricing.get_latest_price("test_s1") == 5.5
    pricing.close()


def test_port_state_transitions(tmp_path: Path) -> None:
    db_path = tmp_path / "stations.sqlite"
    conn = create_db(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("test_s1", "Test Station", 12.95, 77.60, "Operator", "Bangalore", "zone1", "simulated"),
    )
    cursor.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("test_s1_p1", "test_s1", "Type2", 22.0, "Available", "2026-01-01T00:00:00"),
    )
    conn.commit()
    simulator = OccupancySimulator(db_path)
    port_state = simulator.port_states["test_s1_p1"]
    for target in STATUS_TRANSITIONS[port_state.status]:
        simulator._write_status(simulator.connection.cursor(), port_state.port_id, target)
        assert port_state.status == target
    simulator.close()


def test_startup_reset_clears_stale_port_states_and_vehicle_events(tmp_path: Path) -> None:
    db_path = tmp_path / "stations.sqlite"
    conn = create_db(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("test_s1", "Test Station 1", 12.95, 77.60, "Operator", "Bangalore", "zone1", "simulated"),
    )
    cursor.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("test_s2", "Test Station 2", 12.96, 77.61, "Operator", "Bangalore", "zone2", "simulated"),
    )
    cursor.executemany(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("test_s1_p1", "test_s1", "Type2", 22.0, "Charging", "2026-01-01T00:00:00"),
            ("test_s1_p2", "test_s1", "Type2", 22.0, "Reserved", "2026-01-01T00:00:00"),
            ("test_s2_p1", "test_s2", "Type2", 22.0, "Preparing", "2026-01-01T00:00:00"),
            ("test_s2_p2", "test_s2", "Type2", 22.0, "Available", "2026-01-01T00:00:00"),
        ],
    )
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS vehicle_events (id INTEGER PRIMARY KEY AUTOINCREMENT, vehicle_id TEXT, station_id TEXT, port_id TEXT, battery_pct REAL, step INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    cursor.execute(
        "INSERT INTO vehicle_events (vehicle_id, station_id, port_id, battery_pct, step) VALUES (?, ?, ?, ?, ?)",
        ("ev_1", "test_s1", None, 10.0, 5),
    )
    conn.commit()
    conn.close()

    manager = ChargingStationManager(db_path=db_path)
    summary = manager.reset_all_ports_to_available()
    assert summary["station_count"] == 2
    assert summary["total_ports"] == 4
    assert summary["available_ports_before"] == 1
    assert summary["charging_ports_before"] == 3
    assert summary["available_ports_after"] == 4
    assert summary["charging_ports_after"] == 0
    assert summary["ports_reset"] == 3
    assert summary["stale_vehicle_events_cleared"] == 1

    state_1 = manager.get_station_state("test_s1")
    state_2 = manager.get_station_state("test_s2")
    assert state_1 is not None and state_2 is not None
    statuses = [p["status"] for p in state_1["ports"] + state_2["ports"]]
    assert statuses == ["Available", "Available", "Available", "Available"]
    manager.close()
