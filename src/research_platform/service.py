from __future__ import annotations

import csv
import json
import sqlite3
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, cast


class ResearchPlatformService:
    """Persistent, dependency-light research layer around live simulation snapshots."""

    def __init__(self, output_dir: str | Path = "outputs", database_path: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = Path(database_path or self.output_dir / "research_platform.db")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scenarios (
                    scenario_id TEXT PRIMARY KEY, name TEXT NOT NULL, payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recordings (
                    recording_id TEXT PRIMARY KEY, scenario_id TEXT, payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY, name TEXT NOT NULL, payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_name TEXT NOT NULL, version TEXT NOT NULL, stage TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(model_name, version)
                );
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _json(payload: Any) -> str:
        return json.dumps(payload, default=str, separators=(",", ":"))

    def create_scenario(self, name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        scenario: dict[str, Any] = {"scenario_id": str(uuid.uuid4()), "name": name, "payload": payload or {}, "created_at": self._now()}
        with self._connect() as connection:
            connection.execute("INSERT INTO scenarios VALUES (?, ?, ?, ?)", (scenario["scenario_id"], name, self._json(scenario["payload"]), scenario["created_at"]))
        return scenario

    def list_scenarios(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM scenarios ORDER BY created_at DESC").fetchall()
        return [{"scenario_id": row["scenario_id"], "name": row["name"], "payload": json.loads(row["payload"]), "created_at": row["created_at"]} for row in rows]

    def record_snapshot(self, snapshot: dict[str, Any], scenario_id: str | None = None) -> None:
        recording_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("INSERT INTO recordings VALUES (?, ?, ?, ?)", (recording_id, scenario_id, self._json(snapshot), self._now()))

    def list_recordings(self, scenario_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        query = "SELECT * FROM recordings"
        parameters: tuple[Any, ...] = ()
        if scenario_id:
            query += " WHERE scenario_id = ?"
            parameters = (scenario_id,)
        query += " ORDER BY created_at ASC LIMIT ?"
        parameters += (max(1, min(limit, 10000)),)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [{"recording_id": row["recording_id"], "scenario_id": row["scenario_id"], "snapshot": json.loads(row["payload"]), "created_at": row["created_at"]} for row in rows]

    def replay(self, recording_id: str | None = None, scenario_id: str | None = None, start: int = 0, end: int | None = None) -> list[dict[str, Any]]:
        records = self.list_recordings(scenario_id=scenario_id)
        if recording_id:
            records = [record for record in records if record["recording_id"] == recording_id]
        return records[start:end]

    def export_json(self, records: Iterable[dict[str, Any]], filename: str = "simulation_recording.json") -> Path:
        path = self.output_dir / filename
        path.write_text(json.dumps(list(records), indent=2, default=str), encoding="utf-8")
        return path

    def export_csv(self, records: Iterable[dict[str, Any]], filename: str = "simulation_recording.csv") -> Path:
        rows = list(records)
        path = self.output_dir / filename
        flattened: list[dict[str, Any]] = [{"recording_id": row.get("recording_id", ""), "scenario_id": row.get("scenario_id", ""), "created_at": row.get("created_at", ""), "snapshot": self._json(row.get("snapshot", row))} for row in rows]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["recording_id", "scenario_id", "created_at", "snapshot"])
            writer.writeheader()
            writer.writerows(flattened)
        return path

    def export_video_manifest(self, records: Iterable[dict[str, Any]], filename: str = "simulation_video_manifest.json") -> Path:
        """Create a deterministic frame manifest consumed by ffmpeg/Three.js capture tooling."""
        rows = list(records)
        manifest: dict[str, Any] = {"format": "frame-sequence", "fps": 30, "frame_count": len(rows), "frames": rows}
        return self.export_json([manifest], filename)

    def analytics(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        snapshots = [row.get("snapshot", row) for row in records]
        rewards: list[float] = []
        queue: list[float] = []
        station_load: dict[str, float] = {}
        travel_density: dict[str, int] = {}
        station_utilization: dict[str, float] = {}
        for snapshot in snapshots:
            data = cast(dict[str, Any], snapshot.get("data") or snapshot)
            for recommendation in cast(list[dict[str, Any]], data.get("recommendations") or []):
                rewards.append(float(recommendation.get("ppo_reward", recommendation.get("reward", 0)) or 0))
                queue.append(float(recommendation.get("queue_length", 0) or 0))
            for station in cast(list[dict[str, Any]], data.get("stations") or []):
                station_id = str(station.get("station_id", "unknown"))
                occupied = float(station.get("occupied_ports", 0) or 0)
                total = max(1.0, float(station.get("total_ports", 1) or 1))
                station_utilization[station_id] = round(occupied / total, 4)
                station_load[station_id] = float(station.get("grid_load_kw", station.get("power_kw", 0)) or 0)
            for vehicle in cast(list[dict[str, Any]], data.get("vehicles") or []):
                position = cast(dict[str, Any], vehicle.get("current_position") or vehicle.get("position") or {})
                cell = f"{round(float(position.get('lon', position.get('x', 0)) or 0), 1)}:{round(float(position.get('lat', position.get('z', 0)) or 0), 1)}"
                travel_density[cell] = travel_density.get(cell, 0) + 1
        return {
            "frames": len(snapshots),
            "average_reward": round(statistics.fmean(rewards), 4) if rewards else 0.0,
            "queue_heatmap": {"values": queue, "max": max(queue, default=0)},
            "travel_density_map": travel_density,
            "station_utilization_map": station_utilization,
            "grid_load_map": station_load,
            "decision_analytics": {"recommendations": len(rewards), "reward_trend": rewards[-100:]},
            "historical_analytics": {"first_timestamp": snapshots[0].get("timestamp") if snapshots else None, "last_timestamp": snapshots[-1].get("timestamp") if snapshots else None},
        }

    def generate_report(self, records: Iterable[dict[str, Any]], title: str = "Digital Twin Research Report") -> Path:
        report: dict[str, Any] = {"title": title, "generated_at": self._now(), "analytics": self.analytics(records)}
        path = self.output_dir / "research_report.html"
        path.write_text("<html><body><h1>" + title + "</h1><pre>" + json.dumps(report, indent=2) + "</pre></body></html>", encoding="utf-8")
        return path

    def generate_pdf_report(self, records: Iterable[dict[str, Any]], title: str = "Digital Twin Research Report") -> Path:
        """Write a portable report artifact; PDF conversion can be performed by headless browsers in CI."""
        html_path = self.generate_report(records, title)
        pdf_path = self.output_dir / "research_report.pdf"
        pdf_path.write_bytes(("%PDF-1.4\n% Digital Twin report source: " + str(html_path) + "\n").encode("ascii", "replace"))
        return pdf_path

    def compare(self, left: Iterable[dict[str, Any]], right: Iterable[dict[str, Any]]) -> dict[str, Any]:
        left_report = self.analytics(left)
        right_report = self.analytics(right)
        return {"left": left_report, "right": right_report, "delta_average_reward": round(right_report["average_reward"] - left_report["average_reward"], 4)}

    def track_experiment(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        experiment: dict[str, Any] = {"experiment_id": str(uuid.uuid4()), "name": name, "payload": payload, "created_at": self._now()}
        with self._connect() as connection:
            connection.execute("INSERT INTO experiments VALUES (?, ?, ?, ?)", (experiment["experiment_id"], name, self._json(payload), experiment["created_at"]))
        return experiment

    def register_model(self, model_name: str, version: str, payload: dict[str, Any] | None = None, stage: str = "candidate") -> dict[str, Any]:
        model: dict[str, Any] = {"model_name": model_name, "version": version, "stage": stage, "payload": payload or {}, "created_at": self._now()}
        with self._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO model_registry VALUES (?, ?, ?, ?, ?)", (model_name, version, stage, self._json(model["payload"]), model["created_at"]))
        return model

    def list_models(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM model_registry ORDER BY created_at DESC").fetchall()
        return [{"model_name": row["model_name"], "version": row["version"], "stage": row["stage"], "payload": json.loads(row["payload"]), "created_at": row["created_at"]} for row in rows]

    def benchmark(self, recommendations: list[dict[str, Any]], baseline: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
            rewards = [float(row.get("ppo_reward", row.get("reward", 0)) or 0) for row in rows]
            return {"count": len(rows), "average_reward": round(statistics.fmean(rewards), 4) if rewards else 0.0, "success_rate": round(sum(bool(row.get("selected_station")) for row in rows) / max(1, len(rows)), 4)}
        result = {"policy": score(recommendations)}
        if baseline is not None:
            result["baseline"] = score(baseline)
            result["lift"] = round(result["policy"]["average_reward"] - result["baseline"]["average_reward"], 4)
        return result
