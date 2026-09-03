from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Repository:
    """Simple SQLite-backed repository for persistence of recommendations and events."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or "outputs/system.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    selected_station TEXT,
                    ppo_reward REAL,
                    waiting_time REAL,
                    queue_length INTEGER,
                    charging_cost REAL,
                    energy_consumption REAL,
                    payload TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id TEXT,
                    payload TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS charging_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id TEXT,
                    vehicle_id TEXT,
                    payload TEXT
                )
                """
            )
            conn.commit()

    def save_recommendation(self, payload: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO recommendations (selected_station, ppo_reward, waiting_time, queue_length, charging_cost, energy_consumption, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    payload.get("selected_station"),
                    payload.get("ppo_reward"),
                    payload.get("waiting_time"),
                    payload.get("queue_length"),
                    payload.get("charging_cost"),
                    payload.get("energy_consumption"),
                    json.dumps(payload),
                ),
            )
            conn.commit()

    def save_vehicle_history(self, vehicle_id: str, payload: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO vehicle_history (vehicle_id, payload) VALUES (?, ?)", (vehicle_id, json.dumps(payload)))
            conn.commit()

    def save_charging_session(self, station_id: str, vehicle_id: str, payload: dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO charging_sessions (station_id, vehicle_id, payload) VALUES (?, ?, ?)", (station_id, vehicle_id, json.dumps(payload)))
            conn.commit()
