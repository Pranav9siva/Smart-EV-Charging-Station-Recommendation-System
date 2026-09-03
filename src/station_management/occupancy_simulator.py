from __future__ import annotations

import random
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sqlite3 import Connection, connect
from typing import Iterable

from .station_repository import StationRepository

STATUS_TRANSITIONS = {
    "Available": ["Preparing", "Reserved", "Faulted", "Unavailable"],
    "Preparing": ["Charging", "Faulted", "Unavailable"],
    "Charging": ["Finishing", "SuspendedEV", "SuspendedEVSE", "Faulted"],
    "SuspendedEV": ["Charging", "Finishing", "Faulted"],
    "SuspendedEVSE": ["Charging", "Finishing", "Faulted"],
    "Finishing": ["Available", "Faulted"],
    "Reserved": ["Preparing", "Unavailable", "Faulted"],
    "Unavailable": ["Available", "Faulted"],
    "Faulted": ["Available"],
}

CHARGE_RATE_KWH_PER_MIN = 0.5


@dataclass
class PortState:
    port_id: str
    connector_type: str
    power_kw: float
    status: str
    status_updated_at: datetime
    busy_until: datetime | None = None
    queue: deque[datetime] = field(default_factory=deque)

    def is_available(self) -> bool:
        return self.status == "Available"

    def set_status(self, new_status: str) -> None:
        if new_status not in STATUS_TRANSITIONS.get(self.status, []) and new_status != self.status:
            raise ValueError(f"Invalid transition from {self.status} to {new_status}")
        self.status = new_status
        self.status_updated_at = datetime.now(timezone.utc)

    def should_finish(self, now: datetime) -> bool:
        return self.busy_until is not None and now >= self.busy_until

    def next_charge_duration(self, battery_need_kwh: float) -> timedelta:
        duration_min = max(10, int((battery_need_kwh / self.power_kw) * 60))
        return timedelta(minutes=duration_min)


class OccupancySimulator:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.connection = connect(self.db_path)
        self.repo = StationRepository(self.connection)
        self.port_states = self._load_ports()

    def _load_ports(self) -> dict[str, PortState]:
        port_rows = self.repo.get_all_ports()
        return {
            row[0]: PortState(
                port_id=row[0],
                connector_type=row[1],
                power_kw=row[2],
                status=row[3],
                status_updated_at=row[4],
            )
            for row in port_rows
        }

    @contextmanager
    def transaction(self):
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _write_status(self, cursor, port_id: str, new_status: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "UPDATE ports SET status = ?, status_updated_at = ? WHERE port_id = ?",
            (new_status, now, port_id),
        )
        cursor.execute(
            "INSERT INTO occupancy_log (port_id, status, changed_at) VALUES (?, ?, ?)",
            (port_id, new_status, now),
        )
        self.port_states[port_id].status = new_status
        self.port_states[port_id].status_updated_at = datetime.fromisoformat(now)

    def process_arrival(self, station_id: str, battery_need_kwh: float) -> str:
        open_ports = [p for p in self.port_states.values() if p.is_available() and p.port_id.startswith(f"{station_id}_")]
        if not open_ports:
            return "no_available_port"

        port = random.choice(open_ports)
        with self.transaction() as cursor:
            port.set_status("Preparing")
            self._write_status(cursor, port.port_id, "Preparing")
            port.busy_until = datetime.now(timezone.utc) + port.next_charge_duration(battery_need_kwh)
            port.set_status("Charging")
            self._write_status(cursor, port.port_id, "Charging")
        return port.port_id

    def tick(self) -> None:
        now = datetime.now(timezone.utc)
        with self.transaction() as cursor:
            for port_state in self.port_states.values():
                if port_state.status == "Charging" and port_state.should_finish(now):
                    port_state.set_status("Finishing")
                    self._write_status(cursor, port_state.port_id, "Finishing")
                elif port_state.status == "Finishing":
                    port_state.set_status("Available")
                    self._write_status(cursor, port_state.port_id, "Available")
                elif port_state.status in {"Available", "Preparing", "Reserved"}:
                    if random.random() < 0.002:
                        port_state.set_status("Faulted")
                        self._write_status(cursor, port_state.port_id, "Faulted")
                elif port_state.status == "Faulted" and random.random() < 0.05:
                    port_state.set_status("Available")
                    self._write_status(cursor, port_state.port_id, "Available")

    def close(self) -> None:
        self.connection.close()
