# Research-Grade Digital Twin Platform

## Architecture

The existing PPO controller remains the source of recommendations and simulation state. `VisualizationService` receives each controller snapshot, broadcasts it to the existing websocket clients, writes the legacy JSON snapshot, and records the same immutable frame in `ResearchPlatformService`.

The research layer is additive:

- `src/research_platform/service.py`: SQLite-backed scenarios, recordings, replay, exports, analytics, benchmarking, experiments, and model registry.
- `src/research_platform/router.py`: FastAPI research API.
- `src/research_platform/mlflow_adapter.py`: optional MLflow integration with a local fallback.
- `outputs/research_platform.db`: default research database.

## Research workflow

1. Create a scenario with `POST /api/research/scenarios`.
2. Run the unchanged PPO/SUMO simulation.
3. Inspect recorded frames with `GET /api/research/recordings` or replay a time range with `GET /api/research/replay?start=0&end=100`.
4. Query decision analytics and heatmaps with `GET /api/research/analytics`.
5. Compare policies or scenarios with `/policy-comparison`, `/scenario-comparison`, or `/ab-test`.
6. Export JSON, CSV, video frame manifests, HTML reports, or PDF-compatible report artifacts from `/api/research/export/{kind}`.
7. Track experiment metadata and register model versions through `/experiments`, `/mlflow/runs`, and `/models`.

## Compatibility

No PPO environment, action space, reward logic, SUMO integration, or existing dashboard route was replaced. The research service is only called after a snapshot is constructed, so simulations can continue if the research database is unavailable in a deployment-specific adapter.

## Performance

Recordings are append-only SQLite rows. Analytics operates over selected recordings and returns compact heatmap maps. For large studies, use scenario IDs and bounded replay windows, archive completed recordings, and run reports offline through `python -m src.research_platform analytics`.
