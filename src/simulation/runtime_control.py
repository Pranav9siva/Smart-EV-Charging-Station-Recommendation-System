from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SimulationRuntimeControl:
    """Small file-backed command channel shared by the controller and API process."""

    VALID_SPEEDS = (0.5, 1.0, 2.0, 5.0)

    def __init__(self, path: str | Path = "outputs/simulation_control.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {"status": "play", "speed": 1.0, "step_once": False, "reset": False}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                defaults.update(payload)
        except (OSError, json.JSONDecodeError):
            pass
        defaults["status"] = "paused" if defaults.get("status") == "paused" else "play"
        try:
            speed = float(defaults.get("speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        defaults["speed"] = min(self.VALID_SPEEDS, key=lambda value: abs(value - speed))
        defaults["step_once"] = bool(defaults.get("step_once", False))
        defaults["reset"] = bool(defaults.get("reset", False))
        return defaults

    def write(self, *, status: str | None = None, speed: float | None = None, step_once: bool | None = None, reset: bool | None = None) -> dict[str, Any]:
        current = self.read()
        if status is not None:
            current["status"] = "paused" if status == "paused" else "play"
        if speed is not None:
            current["speed"] = min(self.VALID_SPEEDS, key=lambda value: abs(value - float(speed)))
        if step_once is not None:
            current["step_once"] = bool(step_once)
        if reset is not None:
            current["reset"] = bool(reset)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(current, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        return current

    def consume_step(self) -> None:
        self.write(step_once=False)

    def consume_reset(self) -> None:
        self.write(reset=False)
