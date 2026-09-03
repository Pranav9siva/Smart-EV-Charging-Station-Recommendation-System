from __future__ import annotations

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, generate_latest

from src.monitoring.metrics import update_metric
from src.visualization.frontend.app import app


client = TestClient(app)


def test_metrics_endpoint_exposes_prometheus_metrics() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "simulation_steps_total" in body
    assert "api_requests_total" in body


def test_health_endpoint_returns_healthy_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_health_details_returns_expected_sections() -> None:
    response = client.get("/health/details")

    assert response.status_code == 200
    payload = response.json()
    assert payload["database"] in {"healthy", "degraded"}
    assert payload["websocket"] in {"healthy", "degraded"}
    assert payload["simulation"] in {"running", "stopped", "unknown"}
    assert payload["rl_model"] in {"loaded", "unavailable", "unknown"}
    assert "uptime" in payload


def test_metric_updates_are_visible_in_prometheus_output() -> None:
    update_metric("simulation_steps_total", 42)
    update_metric("active_vehicles", 7)
    update_metric("cpu_usage", 33.5)

    response = client.get("/metrics")
    body = response.text

    assert "simulation_steps_total" in body
    assert "active_vehicles" in body
    assert "cpu_usage" in body


def test_prometheus_registry_contains_monitoring_metrics() -> None:
    payload = generate_latest(REGISTRY).decode("utf-8")

    assert "simulation_steps_total" in payload
    assert "api_requests_total" in payload
    assert "websocket_connections" in payload
