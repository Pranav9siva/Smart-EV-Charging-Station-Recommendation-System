"""Serve the `outputs/` folder for the dashboard.

Run this and open http://localhost:8000/dashboard.html
"""
from __future__ import annotations

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.simulation.runtime_control import SimulationRuntimeControl


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/control"):
            self._send_json(self.get_control_state())
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/control"):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                payload = json.loads(body or "{}")
            except json.JSONDecodeError:
                payload = {}
            self._send_json(apply_control_action(payload, control_path=self.server.control_path))
            return
        super().do_POST()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def get_control_state(self) -> dict[str, Any]:
        return apply_control_action({}, control_path=self.server.control_path)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def resolve_control_path(control_path: str | Path | None = None) -> Path:
    if control_path is not None:
        return Path(control_path)
    return ROOT / "outputs" / "simulation_control.json"


def apply_control_action(payload: dict[str, Any] | None = None, control_path: str | Path | None = None) -> dict[str, Any]:
    runtime_control = SimulationRuntimeControl(resolve_control_path(control_path))
    payload = payload or {}
    action = str(payload.get("action") or payload.get("type") or "").strip().lower()
    if action in {"play", "resume"}:
        return runtime_control.write(status="play")
    if action in {"pause", "stop"}:
        return runtime_control.write(status="paused")
    if action == "step_once":
        return runtime_control.write(step_once=True)
    if action == "reset":
        return runtime_control.write(reset=True, status="play")
    if action == "set_speed":
        speed = payload.get("speed")
        try:
            return runtime_control.write(speed=float(speed))
        except (TypeError, ValueError):
            return runtime_control.read()
    return runtime_control.read()


def main() -> None:
    root = ROOT / "outputs"
    if not root.exists():
        print("outputs/ does not exist. Create it by running the simulation or using the Dashboard class to generate initial state.")
        root.mkdir(parents=True, exist_ok=True)
    os.chdir(str(root))
    addr = ("", 8000)
    print(f"Serving {root} at http://localhost:8000/dashboard.html")
    handler = lambda *args, **kwargs: DashboardRequestHandler(*args, directory=str(root), **kwargs)
    server = HTTPServer(addr, handler)
    server.control_path = str(root / "simulation_control.json")
    server.serve_forever()


if __name__ == '__main__':
    main()
