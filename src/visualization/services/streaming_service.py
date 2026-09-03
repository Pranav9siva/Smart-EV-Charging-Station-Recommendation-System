from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from src.research_platform.service import ResearchPlatformService


class VisualizationService:
    """Broadcasts live simulation snapshots to websocket subscribers."""

    def __init__(self, output_dir: str = "outputs", snapshot_record_interval: int = 5) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._subscribers: set[Any] = set()
        self.last_snapshot: dict[str, Any] = {}
        self.snapshot_path = self.output_dir / "live_visualization_state.json"
        # Persist every Nth emitted snapshot by default. Set to 1 to preserve
        # legacy behavior and write every snapshot.
        self.snapshot_record_interval = max(1, int(snapshot_record_interval))
        self._emit_count = 0
        self.research_platform = ResearchPlatformService(output_dir=self.output_dir)

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        self._subscribers.add(websocket)
        self._load_persisted_snapshot()
        initial_payload = self.last_snapshot or self._build_payload({"event": "connected", "data": {}})
        await websocket.send_json(initial_payload)

    async def disconnect(self, websocket: Any) -> None:
        self._subscribers.discard(websocket)

    async def emit_snapshot(self, snapshot: dict[str, Any]) -> None:
        payload = self._build_payload(snapshot)
        self.last_snapshot = payload
        self._write_snapshot(payload)
        self._emit_count += 1
        if self._emit_count % self.snapshot_record_interval == 0:
            self.research_platform.record_snapshot(payload)
        dead: list[Any] = []
        for websocket in list(self._subscribers):
            try:
                await websocket.send_json(payload)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self._subscribers.discard(websocket)

    def emit_snapshot_sync(self, snapshot: dict[str, Any]) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.emit_snapshot(snapshot))
        else:
            asyncio.get_running_loop().create_task(self.emit_snapshot(snapshot))

    def get_last_snapshot(self) -> dict[str, Any]:
        self._load_persisted_snapshot()
        return self.last_snapshot

    def _load_persisted_snapshot(self) -> bool:
        """Load frames written by a controller running outside the API process."""
        try:
            if not self.snapshot_path.exists():
                return False
            payload = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return False
            timestamp = payload.get("timestamp")
            if timestamp == self.last_snapshot.get("timestamp"):
                return False
            self.last_snapshot = self._build_payload(payload)
            return True
        except (OSError, json.JSONDecodeError, TypeError):
            return False

    def _build_payload(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        payload = dict(snapshot or {})
        payload.setdefault("event", "live_step")
        payload.setdefault("timestamp", int(time.time() * 1000))
        payload.setdefault("simulation_time", payload.get("step", payload.get("simulation_time", 0)))
        data = payload.get("data")
        if not isinstance(data, dict):
            data = {}
            payload["data"] = data

        data.setdefault("vehicles", payload.get("vehicles", []))
        station_details = payload.get("station_details", [])
        data.setdefault("stations", station_details)
        data.setdefault("stations_summary", payload.get("stations", {}))
        data.setdefault("station_details", station_details)
        network = data.get("network")
        if not isinstance(network, dict) or not network.get("edges"):
            data["network"] = payload.get("network", {})
        data.setdefault("traffic_lights", payload.get("traffic_lights", []))
        data.setdefault("simulation", payload.get("simulation", {}))
        data.setdefault("kpis", payload.get("kpis", {}))
        data.setdefault("traffic", payload.get("traffic", []))
        data.setdefault("ppo", payload.get("ppo", {}))
        data.setdefault("summary", payload.get("summary", {}))
        data.setdefault("metrics", payload.get("metrics", []))
        data.setdefault("tracked_ids", payload.get("tracked_ids", []))
        data.setdefault("recommendations", payload.get("recommendations", []))
        data.setdefault("latest_recommendation", payload.get("latest_recommendation"))
        data.setdefault("charging_events", payload.get("charging_events", []))
        data.setdefault("xai", payload.get("xai", {}))
        return payload

    def _write_snapshot(self, payload: dict[str, Any]) -> None:
        temporary = self.snapshot_path.with_suffix(".tmp")
        serialized = json.dumps(payload, indent=2)
        temporary.write_text(serialized, encoding="utf-8")
        for _ in range(5):
            try:
                temporary.replace(self.snapshot_path)
                return
            except PermissionError:
                time.sleep(0.02)
        # Windows readers can briefly retain the target file. Keep the frame
        # available rather than dropping the simulation tick on a transient lock.
        self.snapshot_path.write_text(serialized, encoding="utf-8")
        try:
            temporary.unlink()
        except OSError:
            pass
