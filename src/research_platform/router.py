from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.research_platform.service import ResearchPlatformService
from src.research_platform.mlflow_adapter import MLflowTracker

router = APIRouter(prefix="/api/research", tags=["research-platform"])
service = ResearchPlatformService()
mlflow_tracker = MLflowTracker()


@router.get("/scenarios")
def scenarios() -> list[dict[str, Any]]:
    return service.list_scenarios()


@router.post("/scenarios")
def create_scenario(payload: dict[str, Any]) -> dict[str, Any]:
    return service.create_scenario(str(payload.get("name", "unnamed")), payload)


@router.get("/recordings")
def recordings(scenario_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
    return service.list_recordings(scenario_id, limit)


@router.get("/replay")
def replay(scenario_id: str | None = None, start: int = 0, end: int | None = None) -> list[dict[str, Any]]:
    return service.replay(scenario_id=scenario_id, start=start, end=end)


@router.get("/analytics")
def analytics(scenario_id: str | None = None) -> dict[str, Any]:
    return service.analytics(service.list_recordings(scenario_id))


@router.get("/export/{kind}")
def export_recording(kind: str, scenario_id: str | None = None) -> FileResponse:
    records = service.list_recordings(scenario_id)
    if kind == "json":
        path = service.export_json(records)
    elif kind == "csv":
        path = service.export_csv(records)
    elif kind == "video":
        path = service.export_video_manifest(records)
    elif kind == "pdf":
        path = service.generate_pdf_report(records)
    elif kind == "report":
        path = service.generate_report(records)
    else:
        raise HTTPException(status_code=404, detail="Unknown export kind")
    return FileResponse(path)


@router.post("/benchmark")
def benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    return service.benchmark(payload.get("recommendations", []), payload.get("baseline"))


@router.post("/experiments")
def track_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    return service.track_experiment(str(payload.get("name", "experiment")), payload)


@router.get("/models")
def models() -> list[dict[str, Any]]:
    return service.list_models()


@router.post("/models")
def register_model(payload: dict[str, Any]) -> dict[str, Any]:
    return service.register_model(str(payload["model_name"]), str(payload["version"]), payload.get("metadata", payload), str(payload.get("stage", "candidate")))


@router.post("/compare")
def compare(payload: dict[str, Any]) -> dict[str, Any]:
    return service.compare(payload.get("left", []), payload.get("right", []))


@router.post("/scenario-comparison")
def scenario_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    return service.compare(payload.get("scenario_a", []), payload.get("scenario_b", []))


@router.post("/policy-comparison")
def policy_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    return service.benchmark(payload.get("policy", []), payload.get("baseline", []))


@router.post("/ab-test")
def ab_test(payload: dict[str, Any]) -> dict[str, Any]:
    result = service.benchmark(payload.get("variant", []), payload.get("control", []))
    result["experiment"] = payload.get("experiment", "ab-test")
    return result


@router.post("/mlflow/runs")
def mlflow_run(payload: dict[str, Any]) -> dict[str, Any]:
    return mlflow_tracker.log_run(payload.get("parameters", {}), payload.get("metrics", {}), payload.get("tags", {}))
