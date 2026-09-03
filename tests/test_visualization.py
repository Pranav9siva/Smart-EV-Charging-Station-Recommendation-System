from __future__ import annotations

from pathlib import Path

from src.visualization.map.renderer import battery_color, station_status_color, station_status_from_metrics
from src.visualization.services.streaming_service import VisualizationService
from src.simulation.runtime_control import SimulationRuntimeControl


def test_battery_color_and_station_status_helpers() -> None:
    assert battery_color(80) == "#2e8b57"
    assert battery_color(40) == "#f4b400"
    assert battery_color(10) == "#d9534f"
    assert station_status_color("available") == "#2e8b57"
    assert station_status_from_metrics({"available_ports": 2, "charging": False}) == "available"
    assert station_status_from_metrics({"available_ports": 0}) == "busy"


def test_visualization_service_writes_snapshot(tmp_path: Path) -> None:
    service = VisualizationService(output_dir=str(tmp_path))
    service.emit_snapshot_sync({"event": "test", "data": {"vehicles": [], "stations": []}})
    assert (tmp_path / "live_visualization_state.json").exists()


def test_visualization_service_loads_persisted_live_frame(tmp_path: Path) -> None:
    writer = VisualizationService(output_dir=str(tmp_path))
    writer.emit_snapshot_sync({
        "event": "live_step",
        "timestamp": 42,
        "data": {"vehicles": [{"id": "ev_1"}], "stations": [{"station_id": "S1"}]},
    })

    reader = VisualizationService(output_dir=str(tmp_path))
    assert reader._load_persisted_snapshot() is True
    payload = reader.get_last_snapshot()
    assert payload["data"]["vehicles"][0]["id"] == "ev_1"
    assert payload["data"]["stations"][0]["station_id"] == "S1"


def test_runtime_control_supports_observable_commands(tmp_path: Path) -> None:
    control = SimulationRuntimeControl(tmp_path / "simulation_control.json")
    assert control.read()["speed"] == 1.0
    assert control.write(status="paused", speed=5.0, step_once=True)["status"] == "paused"
    assert control.read()["speed"] == 5.0
    control.consume_step()
    assert control.read()["step_once"] is False
