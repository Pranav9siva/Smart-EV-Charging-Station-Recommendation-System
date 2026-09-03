from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class SQLiteStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or "recommendations.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id TEXT NOT NULL,
                    source_node TEXT NOT NULL,
                    destination_node TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def save_recommendation(self, vehicle_id: str, source_node: str, destination_node: str, station_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO recommendations (vehicle_id, source_node, destination_node, station_id) VALUES (?, ?, ?, ?)",
                (vehicle_id, source_node, destination_node, station_id),
            )
            conn.commit()

    def list_recommendations(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT vehicle_id, source_node, destination_node, station_id FROM recommendations ORDER BY id"
            ).fetchall()
        return [
            {
                "vehicle_id": vehicle_id,
                "source_node": source_node,
                "destination_node": destination_node,
                "station_id": station_id,
            }
            for vehicle_id, source_node, destination_node, station_id in rows
        ]
