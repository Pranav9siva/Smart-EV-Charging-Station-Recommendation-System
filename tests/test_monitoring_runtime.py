from __future__ import annotations

from fastapi.testclient import TestClient

from src.monitoring.metrics import get_metrics_payload, update_metric
from src.visualization.frontend.app import app


client = TestClient(app)


def test_runtime_metrics_are_exposed() -> None:
    update_metric("simulation_fps", 42.0)
    update_metric("simulation_step", 100)
    update_metric("episode_reward", 12.5)
    update_metric("charging_queue_length", 3)

    body = client.get("/metrics").text
    assert "simulation_fps" in body
    assert "simulation_step" in body
    assert "episode_reward" in body
    assert "charging_queue_length" in body


def test_grafana_dashboard_json_is_present() -> None:
    import json
    from pathlib import Path

    path = Path("grafana/dashboard.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["title"] == "EV Recommendation Monitoring"
    assert len(payload["panels"]) >= 9


def test_prometheus_config_and_alerts_exist() -> None:
    from pathlib import Path

    assert Path("prometheus.yml").exists()
    assert Path("prometheus_alerts.yml").exists()
