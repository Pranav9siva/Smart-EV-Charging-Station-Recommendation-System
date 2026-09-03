# Research API

All research routes are under `/api/research`.

| Method | Route | Purpose |
|---|---|---|
| GET | `/scenarios` | List scenarios |
| POST | `/scenarios` | Create a scenario |
| GET | `/recordings` | List recorded frames |
| GET | `/replay` | Episode replay and time travel window |
| GET | `/analytics` | Decision, historical, travel-density, station-utilization, grid-load, and queue analytics |
| GET | `/export/json` | JSON export |
| GET | `/export/csv` | CSV export |
| GET | `/export/video` | Deterministic frame manifest for video capture |
| GET | `/export/report` | HTML research report |
| GET | `/export/pdf` | PDF-compatible report artifact |
| POST | `/benchmark` | Policy benchmark and baseline delta |
| POST | `/policy-comparison` | Policy versus baseline comparison |
| POST | `/scenario-comparison` | Scenario comparison |
| POST | `/ab-test` | Control/variant A/B test |
| POST | `/experiments` | Experiment tracking |
| POST | `/mlflow/runs` | MLflow when installed, local fallback otherwise |
| GET/POST | `/models` | Model registry |

FastAPI also exposes interactive OpenAPI documentation at `/docs` and `/redoc`.
