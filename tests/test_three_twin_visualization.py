from __future__ import annotations

from pathlib import Path

from src.visualization.services.road_network_service import RoadNetworkService


def test_road_network_service_builds_payload() -> None:
    service = RoadNetworkService("simulations/bangalore/network.net.xml")
    payload = service.build_payload()
    assert "edges" in payload
    assert "bounds" in payload


def test_three_twin_dashboard_page_exists() -> None:
    page = Path("src/visualization/frontend/three_twin_dashboard.html")
    assert page.exists()


def test_visualization_app_exposes_three_route() -> None:
    from src.visualization.frontend.app import app

    routes = {route.path for route in app.routes}
    assert "/three" in routes


def test_research_dashboard_controls_are_present() -> None:
    page = Path("src/visualization/frontend/three_twin_dashboard.html")
    html = page.read_text(encoding="utf-8")

    for marker in [
        "research-control-center",
        "kpi-grid",
        "vehicle-search",
        "station-search",
        "filters-panel",
        "replay-controls",
        "echarts",
        "xai-panel",
        "xai-selected-station",
        "xai-top-candidates",
    ]:
        assert marker in html
