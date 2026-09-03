# Project Audit: Smart EV Station Recommendation System

## 1. Current architecture

This repository is a research-oriented EV charging digital twin prototype built around:

- SUMO + TraCI for traffic simulation and road-network movement
- Gymnasium + Stable-Baselines3 for RL-based recommendation logic
- SQLite-backed station state and runtime persistence
- A file-backed dashboard plus a FastAPI-based visualization frontend

### Primary runtime layers

- Entry points and orchestration
  - [src/main.py](src/main.py)
  - [scripts/run_simulation.py](scripts/run_simulation.py)
  - [src/__main__.py](src/__main__.py)
  - [scripts/run_dashboard.py](scripts/run_dashboard.py)

- Simulation core
  - [src/simulation/controller.py](src/simulation/controller.py): main orchestrator, SUMO startup, vehicle updates, charging lifecycle, and dashboard payload generation
  - [src/simulation/traci_interface.py](src/simulation/traci_interface.py): SUMO/TRACI startup helpers
  - [src/simulation/runtime_control.py](src/simulation/runtime_control.py): runtime pause/play/reset controls

- EV domain logic
  - [src/ev_management/vehicle_manager.py](src/ev_management/vehicle_manager.py): vehicle fleet generation and tracked-vehicle management
  - [src/ev_model/battery.py](src/ev_model/battery.py): charging/battery calculations

- Station domain logic
  - [src/station_management/manager.py](src/station_management/manager.py): runtime station manager backed by SQLite
  - [src/station_management/station_repository.py](src/station_management/station_repository.py): DB schema and state aggregation
  - [src/station_management/station_state.py](src/station_management/station_state.py): lightweight legacy station manager

- Recommendation and RL
  - [src/recommendation_engine/engine.py](src/recommendation_engine/engine.py): PPO-backed recommendation runtime with heuristic fallback
  - [src/rl_env/gym_ev_charging_env.py](src/rl_env/gym_ev_charging_env.py): Gymnasium environment for station selection
  - [src/rl_env/ev_charging_env.py](src/rl_env/ev_charging_env.py) and [src/rl_env/traci_ev_charging_env.py](src/rl_env/traci_ev_charging_env.py): additional RL variants

- Dashboard and visualization
  - [src/dashboard/dashboard.py](src/dashboard/dashboard.py): writes dashboard JSON/HTML artifacts
  - [src/visualization/services/streaming_service.py](src/visualization/services/streaming_service.py): live snapshot publishing
  - [src/visualization/frontend/app.py](src/visualization/frontend/app.py): FastAPI UI and API routes

## 2. Important files

### Entry points
- [src/main.py](src/main.py): minimal demo entry point for recommendation + RL smoke testing
- [scripts/run_simulation.py](scripts/run_simulation.py): main simulation runner for SUMO-based digital twin execution
- [scripts/run_dashboard.py](scripts/run_dashboard.py): serves dashboard artifacts over HTTP
- [src/__main__.py](src/__main__.py): package-style launcher

### Simulation assets
- SUMO config: [simulations/bangalore/sim.sumocfg](simulations/bangalore/sim.sumocfg)
- Network: [simulations/bangalore/network.net.xml](simulations/bangalore/network.net.xml)
- Routes: [simulations/bangalore/evs.rou.xml](simulations/bangalore/evs.rou.xml)
- Additional station file: [simulations/bangalore/stations.add.xml](simulations/bangalore/stations.add.xml)
- Station DB: [data/stations/stations.sqlite](data/stations/stations.sqlite)
- Station lookup: [data/stations/stations_lookup.json](data/stations/stations_lookup.json)

### RL and recommendation assets
- PPO model path expected by simulation: [runs/ppo_ckpt/ppo_ev_final.zip](runs/ppo_ckpt/ppo_ev_final.zip)
- Related training scripts: [scripts/train_ppo.py](scripts/train_ppo.py)

### Dashboard outputs
- [outputs/dashboard_state.json](outputs/dashboard_state.json)
- [outputs/dashboard_summary.json](outputs/dashboard_summary.json)
- [outputs/live_visualization_state.json](outputs/live_visualization_state.json)
- [outputs/dashboard.html](outputs/dashboard.html)
- [outputs/research_dashboard.html](outputs/research_dashboard.html)

## 3. Execution flow

### Current simulation flow
1. The controller resolves SUMO config, DB path, and model path.
2. It starts SUMO via TraCI and resets station port states.
3. It loads or generates a fleet of EVs and injects them into the SUMO scenario.
4. Each simulation step:
   - advances SUMO
   - updates vehicle positions/battery state
   - detects low-battery vehicles
   - calls the recommendation engine
   - re-routes vehicles toward selected stations
   - processes charging lifecycle and station-port state
   - updates dashboard/visualization outputs
   - records metrics

### Current recommendation flow
1. When a vehicle reaches a low-battery threshold, the controller calls the recommendation engine.
2. The engine attempts PPO inference if a model is available.
3. If loading or inference fails, it falls back to a heuristic station selection.

### Current dashboard flow
1. The controller builds a canonical state object with simulation, KPI, stations, vehicles, traffic, PPO, and recommendation data.
2. The dashboard writer normalizes that payload and writes JSON/HTML artifacts to [outputs](outputs).
3. The visualization service writes live snapshots to [outputs/live_visualization_state.json](outputs/live_visualization_state.json).

## 4. Dependencies

### Runtime dependencies
- Python 3.12 (project metadata targets 3.12)
- SUMO / TraCI
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy / Pandas / NetworkX
- FastAPI / Uvicorn / WebSockets
- SQLite (via Python stdlib)

### Important project configuration
- [pyproject.toml](pyproject.toml)
- [requirements.txt](requirements.txt)
- [SUMO_ENV_SETUP.md](SUMO_ENV_SETUP.md)
- [README_SIMULATION.md](README_SIMULATION.md)

## 5. Simulation components

### SUMO configuration
- The primary Bangalore scenario is present under [simulations/bangalore](simulations/bangalore).
- The controller expects a SUMO config path and launches SUMO with TraCI using the configured network/routes/stations XML files.

### TraCI
- [src/simulation/controller.py](src/simulation/controller.py) contains the main TraCI lifecycle, including startup, stepping, vehicle control, and route changes.
- [src/simulation/traci_interface.py](src/simulation/traci_interface.py) contains a simpler helper wrapper for TraCI startup.

### SimulationController
- The main orchestrator is [src/simulation/controller.py](src/simulation/controller.py).
- It handles:
  - SUMO startup and teardown
  - fleet injection
  - low-battery recommendation handling
  - charging state transitions
  - dashboard refresh and metric collection

### VehicleManager
- [src/ev_management/vehicle_manager.py](src/ev_management/vehicle_manager.py) provides the runtime fleet and tracked-vehicle abstraction.
- It generates synthetic fleets and exposes tracked EV selection.

### StationManager
- [src/station_management/manager.py](src/station_management/manager.py) is the main runtime station manager.
- It uses [src/station_management/station_repository.py](src/station_management/station_repository.py) to read/write station/port state in SQLite.

### Charging logic
- Charging start, wait, progress, completion, and resume flow are implemented inside [src/simulation/controller.py](src/simulation/controller.py).
- Port reservation and release are handled through the station manager.

### Traffic logic
- Traffic is represented through SUMO edge/vehicle state, route information, and dashboard traffic summaries.
- The controller builds traffic payloads from active SUMO vehicles and edge speeds.

## 6. RL components

### Gym environment
- [src/rl_env/gym_ev_charging_env.py](src/rl_env/gym_ev_charging_env.py) defines the main Gymnasium environment for station recommendation.
- It builds observations from tracked vehicles and candidate stations.

### PPO model
- The controller and recommendation engine both look for a PPO model at [runs/ppo_ckpt/ppo_ev_final.zip](runs/ppo_ckpt/ppo_ev_final.zip) by default.
- The engine loads the model when available and uses it for inference.

### Recommendation logic
- [src/recommendation_engine/engine.py](src/recommendation_engine/engine.py) contains the PPO inference path with heuristic fallback.
- The controller uses this engine when a vehicle triggers low-battery handling.

### Reward calculation
- Reward logic is implemented in [src/rl_env/gym_ev_charging_env.py](src/rl_env/gym_ev_charging_env.py).
- It combines negative waiting/distance/cost terms and positive port-availability terms.

## 7. Data

### Station database
- Present at [data/stations/stations.sqlite](data/stations/stations.sqlite)
- Used by the runtime station manager for station metadata, ports, pricing, and occupancy state.

### Station coordinates
- Stored in the station DB and exposed via station snapshots.
- Also referenced by the SUMO station additional file [simulations/bangalore/stations.add.xml](simulations/bangalore/stations.add.xml).

### Tariff / price
- Price history is stored in the station DB and exposed by [src/station_management/station_repository.py](src/station_management/station_repository.py).
- The runtime dashboard uses the station price per kWh in its payloads.

### Ports
- Ports are stored in the database and can be reserved/released by the simulation controller.
- Port status transitions are logged in the SQLite repository.

### Queue
- Queue length is derived from station occupancy and active charging states.
- The controller and dashboard both expose queue-related fields.

### Grid load
- Grid load is derived from active charging port power usage and surfaced in station snapshots and dashboard state.

## 8. Dashboard

### HTML / JavaScript / CSS
- The main dashboard writer is [src/dashboard/dashboard.py](src/dashboard/dashboard.py).
- It emits:
  - [outputs/dashboard.html](outputs/dashboard.html)
  - [outputs/research_dashboard.html](outputs/research_dashboard.html)

### Dashboard state artifacts
- [outputs/dashboard_state.json](outputs/dashboard_state.json)
- [outputs/live_visualization_state.json](outputs/live_visualization_state.json)
- [outputs/dashboard_summary.json](outputs/dashboard_summary.json)

### APIs and state generation
- [src/visualization/frontend/app.py](src/visualization/frontend/app.py) provides browser-facing routes and API endpoints.
- [src/visualization/services/streaming_service.py](src/visualization/services/streaming_service.py) creates a normalized live snapshot payload for websocket/API use.

## 9. Existing tests

The repository already contains a broad test suite under [tests](tests), including:

- [tests/test_main.py](tests/test_main.py)
- [tests/test_prototype.py](tests/test_prototype.py)
- [tests/test_recommendation.py](tests/test_recommendation.py)
- [tests/test_station_management.py](tests/test_station_management.py)
- [tests/test_battery.py](tests/test_battery.py)
- [tests/test_dashboard.py](tests/test_dashboard.py)
- [tests/test_runtime_integration.py](tests/test_runtime_integration.py)
- [tests/test_gym_ev_charging_env.py](tests/test_gym_ev_charging_env.py)
- [tests/test_charging_lifecycle_validation.py](tests/test_charging_lifecycle_validation.py)

These tests cover:
- entry-point smoke runs
- recommendation behavior
- station manager / DB behavior
- battery calculations
- dashboard output generation
- RL environment interfaces
- charging lifecycle interactions

## 10. Existing output files

The outputs directory already contains a number of generated and runtime artifacts:

- [outputs/dashboard_state.json](outputs/dashboard_state.json)
- [outputs/dashboard_summary.json](outputs/dashboard_summary.json)
- [outputs/live_visualization_state.json](outputs/live_visualization_state.json)
- [outputs/dashboard.html](outputs/dashboard.html)
- [outputs/research_dashboard.html](outputs/research_dashboard.html)
- [outputs/simulation_metrics.csv](outputs/simulation_metrics.csv)
- [outputs/recommendation_log.csv](outputs/recommendation_log.csv)
- [outputs/charging_events.csv](outputs/charging_events.csv)
- [outputs/evaluation_summary.csv](outputs/evaluation_summary.csv)
- [outputs/latest_explanation.json](outputs/latest_explanation.json)
- [outputs/research_platform.db](outputs/research_platform.db)

## 11. Duplicate, obsolete, generated, temporary, debug, and unused files

### Likely duplicates or overlapping implementations
- [src/station_management/station_state.py](src/station_management/station_state.py) and [src/station_management/manager.py](src/station_management/manager.py) both provide station-management behavior, but one is a lightweight legacy abstraction and the other is the runtime-backed implementation.
- [src/rl_env/ev_charging_env.py](src/rl_env/ev_charging_env.py), [src/rl_env/gym_ev_charging_env.py](src/rl_env/gym_ev_charging_env.py), and [src/rl_env/traci_ev_charging_env.py](src/rl_env/traci_ev_charging_env.py) overlap conceptually and should be consolidated later if the project is streamlined.
- Several scripts under [scripts](scripts) appear to be alternatives or diagnostic entry points, including profiling and validation scripts, rather than one canonical workflow.

### Suspicious or likely temporary/debug files
- [scripts/diagnostics](scripts/diagnostics) contains diagnostic helpers that are likely not part of the canonical runtime path.
- [scripts/profile_1000_short.py](scripts/profile_1000_short.py), [scripts/profile_refresh_dashboard_only.py](scripts/profile_refresh_dashboard_only.py), and [scripts/profile_simulation_fast.py](scripts/profile_simulation_fast.py) appear to be profiling/debugging utilities rather than primary entry points.
- [scripts/test_traci_env.py](scripts/test_traci_env.py) looks like an environment-specific smoke test.
- [outputs/archive](outputs/archive) contains archived runtime artifacts, which is reasonable but should remain clearly separated.

### Generated files that are expected to exist at runtime
- [outputs/dashboard_state.json](outputs/dashboard_state.json)
- [outputs/live_visualization_state.json](outputs/live_visualization_state.json)
- [outputs/research_platform.db](outputs/research_platform.db)
- [outputs/simulation_metrics.csv](outputs/simulation_metrics.csv)
- [outputs/recommendation_log.csv](outputs/recommendation_log.csv)

### Potentially missing or fragile components for a single-command digital twin
- A single canonical CLI wrapper is partially present but still dispersed across multiple scripts.
- The repository has multiple “entry-point-like” scripts and could benefit from a single orchestrated command path.
- The PPO model file path is expected but the repo should clearly declare whether the model is optional or required for startup.
- The dashboard data contract is present, but some frontends may still rely on older field names and need normalization for a single UI experience.

## 12. Dashboard data currently available

The current runtime already produces or exposes:

- simulation time/step/running state
- fleet size / EV count / tracked count
- average speed / traffic density / wait time / energy consumed
- station totals, availability, occupancy, and queue totals
- per-vehicle position, battery, route, recommendation, cost, and reward data
- per-station availability, queue, price, wait, grid load, and charging state
- recommendation log and charging events
- latest explanation payload for XAI output

## 13. Dashboard data currently missing or weak

The following areas appear incomplete or inconsistent for a clean single-command digital twin experience:

- A single canonical dashboard schema is partially present but still coexists with older legacy field names.
- Some UI consumers may expect a more uniform contract across vehicles, stations, traffic, PPO, and XAI data.
- The repository has several dashboard-related artifacts, but the “single-command” UX is not yet fully unified around one entry point and one data contract.
- Some outputs are produced conditionally or only after a full simulation run, so the first-run experience is not yet fully polished.

## 14. Recommended final architecture

For a reliable single-command EV Digital Twin experience, the project should converge around:

1. One canonical entry point
   - A single script or CLI command that launches the simulation, dashboard, and visualization path.

2. One canonical state contract
   - A shared payload schema for vehicles, stations, traffic, metrics, PPO summary, and XAI state.
   - Compatibility fallbacks should remain in place for older frontends, but the canonical shape should be authoritative.

3. One canonical station state source
   - Keep [src/station_management/manager.py](src/station_management/manager.py) as the runtime-backed implementation.
   - Keep the DB as the authoritative source of station metadata, ports, price, and occupancy.

4. One canonical recommendation path
   - Keep PPO inference when a model is present.
   - Keep heuristic fallback when no model is available.
   - Make the model-availability behavior explicit in the CLI and logs.

5. One canonical dashboard pipeline
   - Generate dashboard state and HTML from a single shared path.
   - Write outputs to [outputs](outputs) in a stable structure.
   - Keep the data writer and frontend consumer aligned.

6. One canonical runtime configuration
   - Use a single config/object for SUMO, fleet size, tracked vehicles, station DB, model path, and dashboard/report intervals.

## A. What already works

- The main repository structure is coherent and the core modules are present.
- SUMO/TraCI startup and simulation orchestration are implemented in [src/simulation/controller.py](src/simulation/controller.py).
- The recommendation engine and RL environment are implemented and wired to the controller.
- The station repository and runtime station manager are functional and use SQLite.
- The dashboard writer and live visualization service already produce real output artifacts.
- Existing tests cover significant portions of the main runtime, recommendation logic, station behavior, battery logic, and dashboard generation.

## B. What is broken or fragile

- The project still has multiple overlapping entry points and workflow styles, which makes “single-command” operation less straightforward than it should be.
- The codebase contains overlapping station-management and RL implementations that add maintenance complexity.
- Some dashboard/front-end paths appear to rely on a mix of legacy and canonical field names, which can create inconsistencies.
- Some runtime assets are optional or loosely declared, so the experience can vary depending on whether SUMO, a model file, or a station DB is present.
- The full simulation experience is not yet unified behind one clean CLI contract.

## C. What must be changed

- Consolidate the user-facing workflow around one canonical entry point.
- Standardize the state payload between controller, dashboard, and visualization service.
- Reduce overlap between legacy and runtime station-management implementations.
- Make model availability, SUMO availability, and DB availability explicit and user-friendly.
- Keep the runtime behavior intact while simplifying the operational path.

## D. What should NOT be changed

- The existing SUMO, TraCI, station, recommendation, and RL core modules should not be replaced wholesale.
- The current DB-backed station storage and simulation lifecycle should be preserved.
- The existing dashboard artifacts and output conventions should be preserved unless a compatibility issue requires a small, targeted adjustment.
- The repository should not be re-implemented from scratch; the focus should be consolidation and reliability rather than redesign.
