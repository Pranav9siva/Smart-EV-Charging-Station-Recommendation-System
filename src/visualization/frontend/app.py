from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi import Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.monitoring.health import router as health_router
from src.monitoring.middleware import register_monitoring_middleware
from src.monitoring.metrics import REGISTRY as MONITORING_REGISTRY
from src.research_platform.router import router as research_router
from src.visualization.websocket.server import router as ws_router
from src.simulation.runtime_control import SimulationRuntimeControl

app = FastAPI(title="EV Visualization")
register_monitoring_middleware(app)
app.include_router(ws_router)
app.include_router(health_router)
app.include_router(research_router)
runtime_control = SimulationRuntimeControl()

frontend_dir = Path(__file__).resolve().parent
repo_root = Path(__file__).resolve().parents[3]
explanation_file = repo_root / "outputs" / "latest_explanation.json"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/live")
def live() -> FileResponse:
    return FileResponse(frontend_dir / "live_dashboard.html")


@app.get("/three")
def three_twin() -> FileResponse:
    return FileResponse(frontend_dir / "three_twin_dashboard.html")


@app.get("/api/simulation/status")
def simulation_status() -> dict:
    return runtime_control.read()


@app.post("/api/simulation/control")
def simulation_control(command: dict = Body(...)) -> dict:
    action = str(command.get("action", "")).lower()
    if action == "play":
        return runtime_control.write(status="play")
    if action == "pause":
        return runtime_control.write(status="paused")
    if action == "reset":
        return runtime_control.write(status="paused", reset=True, step_once=False)
    if action == "step":
        return runtime_control.write(status="paused", step_once=True)
    if action == "speed":
        return runtime_control.write(speed=float(command.get("value", 1.0)))
    return {"error": "action must be play, pause, reset, step, or speed"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(MONITORING_REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/explanations/latest")
def latest_explanation() -> JSONResponse:
    if explanation_file.exists():
        return JSONResponse(json.loads(explanation_file.read_text(encoding="utf-8")))
    return JSONResponse({})


@app.get("/api/explanations/export")
def export_explanation() -> FileResponse:
    if explanation_file.exists():
        return FileResponse(explanation_file)
    return FileResponse(frontend_dir / "index.html")
