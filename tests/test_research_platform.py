from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.research_platform.service import ResearchPlatformService
from src.visualization.frontend.app import app


def sample_snapshot(step: int, reward: float = 1.0) -> dict:
    return {
        "step": step,
        "timestamp": step,
        "data": {
            "vehicles": [{"id": "ev-1", "current_position": {"lon": step, "lat": step}}],
            "stations": [{"station_id": "S1", "occupied_ports": 1, "total_ports": 2, "grid_load_kw": 20}],
            "recommendations": [{"selected_station": "S1", "ppo_reward": reward, "queue_length": step}],
        },
    }


def test_recording_exports_and_analytics(tmp_path: Path) -> None:
    service = ResearchPlatformService(tmp_path)
    service.record_snapshot(sample_snapshot(1))
    service.record_snapshot(sample_snapshot(2, 2.0))
    records = service.list_recordings()

    assert len(records) == 2
    assert service.analytics(records)["average_reward"] == 1.5
    assert service.export_csv(records).exists()
    assert service.export_json(records).exists()
    assert service.export_video_manifest(records).exists()
    assert service.generate_report(records).exists()


def test_research_api_supports_scenarios_and_benchmark() -> None:
    client = TestClient(app)
    response = client.post("/api/research/scenarios", json={"name": "baseline", "seed": 7})
    assert response.status_code == 200
    assert response.json()["name"] == "baseline"

    response = client.post("/api/research/benchmark", json={"recommendations": [{"selected_station": "S1", "ppo_reward": 1}], "baseline": [{"selected_station": "S1", "ppo_reward": 0}]})
    assert response.status_code == 200
    assert response.json()["lift"] == 1.0
