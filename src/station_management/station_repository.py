from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sqlite3 import Connection, connect
from typing import Any

from .pricing import current_tariff_period

EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


class StationRepository:
    def __init__(self, connection: Connection | str | Path):
        if isinstance(connection, (str, Path)):
            db_path = Path(connection)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            if db_path.exists() and db_path.stat().st_size > 0:
                try:
                    test_connection = connect(str(db_path))
                    test_connection.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                    test_connection.close()
                    connection = connect(str(db_path))
                except sqlite3.DatabaseError:
                    db_path.unlink(missing_ok=True)
                    connection = connect(str(db_path))
            else:
                connection = connect(str(db_path))
            self.connection = connection
            self._ensure_schema()
        else:
            self.connection = connection
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.connection.execute(
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
        self.connection.execute(
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
        self.connection.execute(
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
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS occupancy_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                port_id TEXT,
                status TEXT,
                changed_at TEXT
            )
            """
        )
        self.connection.commit()

    def get_all_stations(self) -> list[tuple[str, str, float, float, str, str, str, str]]:
        cursor = self.connection.execute(
            """
            SELECT station_id, name, lat, lon, operator, address, grid_zone_id, data_source
            FROM stations
            WHERE station_id NOT LIKE 'sim_station_%'
                AND station_id NOT LIKE 'phase1c_%'
                AND COALESCE(station_id, '') != ''
            """
        )
        return cursor.fetchall()

    def get_all_ports(self) -> list[tuple[str, str, float, str, str, str]]:
        cursor = self.connection.execute("SELECT port_id, connector_type, power_kw, status, status_updated_at, station_id FROM ports")
        return cursor.fetchall()

    def get_nearby_stations(self, lat: float, lon: float, radius_km: float) -> list[dict[str, Any]]:
        stations = []
        for row in self.get_all_stations():
            distance = haversine(lat, lon, row[2], row[3])
            if distance <= radius_km:
                stations.append(
                    {
                        "station_id": row[0],
                        "name": row[1],
                        "lat": row[2],
                        "lon": row[3],
                        "operator": row[4],
                        "address": row[5],
                        "grid_zone_id": row[6],
                        "data_source": row[7],
                        "distance_km": round(distance, 3),
                    }
                )
        return sorted(stations, key=lambda s: s["distance_km"])

    def get_station_state(self, station_id: str) -> dict[str, Any] | None:
        cursor = self.connection.execute("SELECT station_id, name, lat, lon, operator, address, grid_zone_id, data_source FROM stations WHERE station_id = ?", (station_id,))
        station_row = cursor.fetchone()
        if not station_row:
            return None
        if str(station_row[0]).startswith("sim_station_") or str(station_row[0]).startswith("phase1c_"):
            return None

        ports = self.connection.execute(
            "SELECT port_id, connector_type, power_kw, status, status_updated_at FROM ports WHERE station_id = ?",
            (station_id,),
        ).fetchall()

        price = self.connection.execute(
            "SELECT price_per_kwh FROM price_history WHERE station_id = ? ORDER BY recorded_at DESC LIMIT 1",
            (station_id,),
        ).fetchone()

        return {
            "station_id": station_row[0],
            "name": station_row[1],
            "lat": station_row[2],
            "lon": station_row[3],
            "operator": station_row[4],
            "address": station_row[5],
            "data_source": station_row[7],
            "ports": [
                {
                    "port_id": port[0],
                    "connector_type": port[1],
                    "power_kw": port[2],
                    "status": port[3],
                    "status_updated_at": port[4],
                }
                for port in ports
            ],
            "price_per_kwh": price[0] if price else None,
        }

    def get_occupancy_history(self, port_id: str, window: timedelta) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - window
        rows = self.connection.execute(
            "SELECT status, changed_at FROM occupancy_log WHERE port_id = ? AND changed_at >= ? ORDER BY changed_at DESC",
            (port_id, cutoff.isoformat()),
        ).fetchall()
        return [{"status": row[0], "changed_at": row[1]} for row in rows]

    def log_status_change(self, port_id: str, new_status: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            "UPDATE ports SET status = ?, status_updated_at = ? WHERE port_id = ?",
            (new_status, now, port_id),
        )
        self.connection.execute(
            "INSERT INTO occupancy_log (port_id, status, changed_at) VALUES (?, ?, ?)",
            (port_id, new_status, now),
        )
        self.connection.commit()

    def get_station_aggregate(self, station_id: str) -> dict[str, Any]:
        ports = self.connection.execute("SELECT port_id, status, power_kw FROM ports WHERE station_id = ?", (station_id,)).fetchall()
        available = sum(1 for row in ports if row[1] == "Available")
        occupied = sum(1 for row in ports if row[1] in {"Charging", "Preparing", "SuspendedEV", "SuspendedEVSE", "Finishing"})
        historical = self.connection.execute(
            "SELECT status, changed_at FROM occupancy_log WHERE port_id IN (SELECT port_id FROM ports WHERE station_id = ?) ORDER BY changed_at DESC LIMIT 50",
            (station_id,),
        ).fetchall()
        return {
            "available_ports": available,
            "occupied_ports": occupied,
            "avg_wait_estimate": self._estimate_wait_time(ports, historical),
        }

    def get_station_id_for_port(self, port_id: str) -> str:
        row = self.connection.execute("SELECT station_id FROM ports WHERE port_id = ?", (port_id,)).fetchone()
        return str(row[0]) if row and row[0] is not None else ""

    def get_station_snapshots(self) -> list[dict[str, Any]]:
        """Return station state + aggregate metrics in a single bulk read."""
        station_rows = self.connection.execute(
            """
            SELECT station_id, name, lat, lon, operator, address, grid_zone_id, data_source
            FROM stations
            WHERE station_id NOT LIKE 'sim_station_%'
                AND station_id NOT LIKE 'phase1c_%'
                AND COALESCE(station_id, '') != ''
            """
        ).fetchall()
        port_rows = self.connection.execute(
            "SELECT port_id, station_id, connector_type, power_kw, status, status_updated_at FROM ports"
        ).fetchall()
        price_rows = self.connection.execute(
            """
            SELECT p.station_id, p.price_per_kwh
            FROM price_history p
            INNER JOIN (
                SELECT station_id, MAX(recorded_at) AS latest_recorded_at
                FROM price_history
                GROUP BY station_id
            ) latest ON latest.station_id = p.station_id AND latest.latest_recorded_at = p.recorded_at
            """
        ).fetchall()

        price_by_station = {str(row[0]): row[1] for row in price_rows}
        ports_by_station: dict[str, list[dict[str, Any]]] = {}
        for row in port_rows:
            sid = str(row[1])
            ports_by_station.setdefault(sid, []).append(
                {
                    "port_id": row[0],
                    "connector_type": row[2],
                    "power_kw": row[3],
                    "status": row[4],
                    "status_updated_at": row[5],
                }
            )

        occupied_statuses = {"Charging", "Preparing", "SuspendedEV", "SuspendedEVSE", "Finishing"}
        snapshots: list[dict[str, Any]] = []
        for row in station_rows:
            station_id = str(row[0])
            ports = ports_by_station.get(station_id, [])
            available_ports = sum(1 for port in ports if port.get("status") == "Available")
            occupied_ports = sum(1 for port in ports if port.get("status") in occupied_statuses)
            total_ports = len(ports)
            charging_ports = [port for port in ports if port.get("status") == "Charging"]
            queue_length = max(0, occupied_ports - len(charging_ports))
            avg_wait_estimate = 0.0
            if charging_ports:
                avg_wait_estimate = round(
                    sum(float(port.get("power_kw", 0.0) or 0.0) for port in charging_ports) / len(charging_ports) * 0.5,
                    2,
                )
            grid_load_kw = sum(float(port.get("power_kw", 0.0) or 0.0) for port in charging_ports)
            occupancy_rate = (occupied_ports / total_ports) if total_ports else 0.0
            if charging_ports:
                charging_state = "charging"
            elif queue_length > 0:
                charging_state = "queued"
            else:
                charging_state = "idle"

            snapshots.append(
                {
                    "station_id": station_id,
                    "name": row[1],
                    "lat": row[2],
                    "lon": row[3],
                    "operator": row[4],
                    "address": row[5],
                    "grid_zone_id": row[6],
                    "data_source": row[7],
                    "price_per_kwh": price_by_station.get(station_id),
                    "ports": ports,
                    "connector_types": sorted({str(port.get("connector_type")) for port in ports if port.get("connector_type")}),
                    "available_ports": available_ports,
                    "occupied_ports": occupied_ports,
                    "charging_ports": len(charging_ports),
                    "total_ports": total_ports,
                    "queue_length": queue_length,
                    "avg_wait_estimate": avg_wait_estimate,
                    "grid_load_kw": grid_load_kw,
                    "occupancy_rate": occupancy_rate,
                    "occupancy_percent": round(occupancy_rate * 100.0, 2),
                    "charging_state": charging_state,
                }
            )

        return snapshots

    def get_port_status_counts(self) -> dict[str, int]:
        rows = self.connection.execute("SELECT status, COUNT(*) FROM ports GROUP BY status").fetchall()
        counts = {str(status): int(total) for status, total in rows}
        total_ports = int(sum(counts.values()))
        available_ports = int(counts.get("Available", 0))
        charging_like_statuses = {"Charging", "Preparing", "SuspendedEV", "SuspendedEVSE", "Finishing", "Reserved"}
        charging_ports = int(sum(counts.get(status, 0) for status in charging_like_statuses))
        station_count_row = self.connection.execute("SELECT COUNT(*) FROM stations").fetchone()
        station_count = int(station_count_row[0]) if station_count_row else 0
        return {
            "station_count": station_count,
            "total_ports": total_ports,
            "available_ports": available_ports,
            "charging_ports": charging_ports,
        }

    def reset_ports_for_new_simulation(self) -> dict[str, int]:
        """Reset transient runtime state so a new simulation run starts cleanly."""
        before = self.get_port_status_counts()
        now = datetime.now(timezone.utc).isoformat()
        changed_port_rows = self.connection.execute(
            "SELECT port_id FROM ports WHERE status != ?",
            ("Available",),
        ).fetchall()
        changed_port_ids = [str(row[0]) for row in changed_port_rows]

        # Keep station topology/metadata intact; only clear transient runtime state.
        with self.connection:
            changed_ports = self.connection.execute(
                "UPDATE ports SET status = ?, status_updated_at = ? WHERE status != ?",
                ("Available", now, "Available"),
            ).rowcount

            # Preserve historical occupancy by logging resets for ports that were not Available.
            if changed_port_ids:
                self.connection.executemany(
                    "INSERT INTO occupancy_log (port_id, status, changed_at) VALUES (?, ?, ?)",
                    [(port_id, "Available", now) for port_id in changed_port_ids],
                )

            # `vehicle_events` is a transient per-run diagnostic table created by the controller.
            row = self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vehicle_events'"
            ).fetchone()
            cleared_vehicle_events = 0
            if row:
                cleared_vehicle_events = self.connection.execute("DELETE FROM vehicle_events").rowcount

        after = self.get_port_status_counts()
        return {
            "station_count": after["station_count"],
            "total_ports": after["total_ports"],
            "available_ports_before": before["available_ports"],
            "charging_ports_before": before["charging_ports"],
            "available_ports_after": after["available_ports"],
            "charging_ports_after": after["charging_ports"],
            "ports_reset": int(changed_ports or 0),
            "stale_vehicle_events_cleared": int(cleared_vehicle_events or 0),
        }

    def _estimate_wait_time(self, ports: Iterable[tuple[str, str, float]], historical: list[tuple[str, str]]) -> float:
        active_charges = [row for row in ports if row[1] == "Charging"]
        if not active_charges:
            return 0.0
        return round(sum(row[2] for row in active_charges) / len(active_charges) * 0.5, 2)

    def close(self) -> None:
        self.connection.close()
