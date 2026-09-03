import sqlite3

from src.station_management.manager import ChargingStationManager


def _build_test_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE stations (station_id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL, operator TEXT, address TEXT, grid_zone_id TEXT, data_source TEXT)"
    )
    conn.execute(
        "CREATE TABLE ports (port_id TEXT PRIMARY KEY, station_id TEXT, connector_type TEXT, power_kw REAL, status TEXT, status_updated_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE occupancy_log (id INTEGER PRIMARY KEY AUTOINCREMENT, port_id TEXT, status TEXT, changed_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE price_history (id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT, price_per_kwh REAL, tariff_period TEXT, recorded_at TEXT)"
    )

    conn.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("st_1", "Station 1", 12.0, 77.0, "Test", "Addr", "zone", "sim"),
    )
    conn.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("st_2", "Station 2", 12.1, 77.1, "Test", "Addr", "zone", "sim"),
    )

    conn.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("st_1_p1", "st_1", "CCS", 30.0, "Available", "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("st_1_p2", "st_1", "CCS", 60.0, "Charging", "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("st_2_p1", "st_2", "CCS", 22.0, "Charging", "2026-01-01T00:00:00+00:00"),
    )

    conn.execute(
        "INSERT INTO price_history (station_id, price_per_kwh, tariff_period, recorded_at) VALUES (?, ?, ?, ?)",
        ("st_1", 0.25, "non_solar_hour", "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO price_history (station_id, price_per_kwh, tariff_period, recorded_at) VALUES (?, ?, ?, ?)",
        ("st_2", 0.30, "non_solar_hour", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    return conn


def test_station_snapshots_include_metrics_and_ports() -> None:
    conn = _build_test_db()
    manager = ChargingStationManager(connection=conn)

    snapshots = manager.get_station_snapshots()
    assert len(snapshots) == 2

    by_id = {row["station_id"]: row for row in snapshots}
    st1 = by_id["st_1"]
    st2 = by_id["st_2"]

    assert st1["available_ports"] == 1
    assert st1["occupied_ports"] == 1
    assert st1["total_ports"] == 2
    assert st1["charging_ports"] == 1
    assert st1["queue_length"] == 0
    assert st1["price_per_kwh"] == 0.25
    assert st1["grid_load_kw"] == 60.0
    assert st1["charging_state"] == "charging"
    assert st1["occupancy_percent"] == 50.0
    assert st1["connector_types"] == ["CCS"]
    assert len(st1["ports"]) == 2

    assert st2["available_ports"] == 0
    assert st2["occupied_ports"] == 1
    assert st2["total_ports"] == 1
    assert st2["charging_ports"] == 1
    assert st2["queue_length"] == 0
    assert st2["price_per_kwh"] == 0.30
    assert st2["grid_load_kw"] == 22.0
    assert st2["charging_state"] == "charging"

    conn.close()


def test_get_station_metrics_uses_snapshot_shape() -> None:
    conn = _build_test_db()
    manager = ChargingStationManager(connection=conn)

    metrics = manager.get_station_metrics("st_1")
    assert metrics is not None
    assert metrics["station_id"] == "st_1"
    assert metrics["available_ports"] == 1
    assert metrics["occupied_ports"] == 1
    assert metrics["total_ports"] == 2
    assert metrics["charging_ports"] == 1
    assert metrics["queue_length"] == 0
    assert metrics["price_per_kwh"] == 0.25
    assert metrics["power_kw"] == 60.0
    assert metrics["charging_state"] == "charging"

    conn.close()


def test_station_snapshot_updates_after_reservation_and_release() -> None:
    conn = _build_test_db()
    manager = ChargingStationManager(connection=conn)

    initial = manager.get_station_snapshot("st_1")
    assert initial is not None
    assert initial["available_ports"] == 1
    assert initial["occupied_ports"] == 1

    port_id = manager.reserve_port("st_1")
    assert port_id == "st_1_p1"

    reserved = manager.get_station_snapshot("st_1", force_refresh=True)
    assert reserved is not None
    assert reserved["available_ports"] == 0
    assert reserved["occupied_ports"] == 2
    assert reserved["charging_ports"] == 2
    assert reserved["queue_length"] == 0
    assert reserved["charging_state"] == "charging"

    manager.release_port(port_id)
    released = manager.get_station_snapshot("st_1", force_refresh=True)
    assert released is not None
    assert released["available_ports"] == 1
    assert released["occupied_ports"] == 1
    assert released["charging_ports"] == 1
    assert released["queue_length"] == 0
    assert released["charging_state"] == "charging"

    conn.close()
