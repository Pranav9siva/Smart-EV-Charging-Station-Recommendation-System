# Project Structure

## Goals of the Cleanup

- Preserve simulation, PPO, SUMO, dashboard, and reproducibility behavior.
- Remove root-level noise and cache artifacts.
- Separate active outputs from historical diagnostics.
- Keep uncertain files in review/archive instead of deleting blindly.

## Top-Level Folders

- `src/`: application source code.
- `scripts/`: executable entry scripts for simulation, training, setup, and utilities.
- `simulations/`: SUMO-ready scenario files (`.sumocfg`, `.net.xml`, `.rou.xml`, `.add.xml`).
- `sumo/`: network/build assets and OSM-derived resources.
- `data/`: input datasets, station DB, fleet and OSM files.
- `runs/`: PPO checkpoints and final models.
- `outputs/`: generated artifacts, split into active and archived sections.
- `docs/`: user, deployment, research, and architecture docs.
- `tests/`: regression, integration, and validation tests.
- `archive/review/`: uncertain, temporary, and duplicate candidates retained for review.

## Source Layout (`src/`)

- `src/simulation/`
  - `controller.py`: primary runtime orchestrator.
  - `runtime_control.py`: pause/play/speed/reset control contract.
  - `traci_interface.py`: SUMO TraCI integration helpers.
  - `bangalore_scenario.py`: scenario generation/prepare helpers.
- `src/ev_management/`: EV lifecycle and state tracking.
- `src/station_management/`: station state, occupancy, pricing, repository access.
- `src/recommendation_engine/`: PPO inference-time recommendation engine.
- `src/recommendation/`: heuristic recommendation and candidate builder helpers.
- `src/rl_env/`: Gym and TraCI RL environments, PPO training modules.
- `src/visualization/`: dashboard/frontend assets, websocket services.
- `src/xai/`: explainability generation and export.
- `src/research_platform/`: experiment recording and API routes.
- `src/monitoring/`: health checks and metrics instrumentation.
- `src/api/`: API entrypoint and routers.

## Entry Points

- Simulation: `scripts/run_simulation.py`
- PPO training: `scripts/train_ppo.py`
- Dashboard static server: `scripts/run_dashboard.py`
- Visualization API/UI: `src/visualization/frontend/app.py`
- Basic demo entry: `src/main.py` and `src/__main__.py`

## Output Layout (`outputs/`)

- Root remains runtime-compatible for existing code:
  - `dashboard_state.json`, `simulation_metrics.csv`, etc.
- Organized mirrors:
  - `outputs/latest/`: latest CSV/JSON snapshots.
  - `outputs/dashboards/`: dashboard HTML copies.
  - `outputs/visualizations/`: plot/image copies.
  - `outputs/explainability/shap/`, `outputs/explainability/lime/`.
- Historical artifacts:
  - `outputs/archive/<timestamp>/phase3/`
  - `outputs/archive/<timestamp>/diagnostics/`

## Safety Rules Applied

- No core SUMO, PPO, station DB, or source modules were deleted.
- Temporary diagnostics were moved to `archive/review/`.
- Deterministic cache artifacts (`__pycache__`, `.pyc`, `.pytest_cache`) were removed.
- Cleanup actions are recorded in `docs/architecture/cleanup_manifest.json`.

## Inventory Files

- `docs/architecture/file_inventory_project.csv`
- `docs/architecture/file_inventory_project_summary.json`

These files include per-file classification and relocation recommendations.
