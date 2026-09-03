# AGENTS

This repository is a Python 3.12 prototype for EV charging station recommendation and simulation.
It is built around SUMO + TraCI for traffic simulation, Gymnasium + Stable-Baselines3 for reinforcement learning,
and SQLite/SQLAlchemy for persistence.

## Primary guidance for AI coding agents

- Use `pyproject.toml` and `requirements.txt` as the authoritative dependency and runtime configuration.
- Prefer `README.md` for project intent and high-level structure.
- Use `SUMO_ENV_SETUP.md` and `README_SIMULATION.md` for SUMO installation, environment setup, and simulation guidance.
- The repository is not a web application; focus on simulation, recommendation logic, and RL environment code.

## Main source areas

- `src/` — core application packages
  - `src/simulation/` — SUMO / TraCI integration and scenario management
  - `src/rl_env/` — Gymnasium environment, reward logic, and policy training code
  - `src/recommendation_engine/` — station recommendation and ranking logic
  - `src/station_management/` — station state, capacity, port availability, and pricing behavior
  - `src/route_planning/` — network graph and travel-time modeling
  - `src/storage/` — SQLite persistence and recommendation history
- `scripts/` — runnable helper scripts for network generation, simulation runs, and station placement
- `tests/` — pytest-based unit/integration tests

## Run and test conventions

- Run tests with `python -m pytest tests` or `pytest tests`.
- Run the main demo with `python src/main.py`.
- Use `scripts/run_traci.py`, `scripts/run_traci_bangalore.py`, `scripts/run_traci_recommend.py`, and `scripts/run_traci_rl.py` for SUMO-based simulation and reinforcement learning workflows.
- The `src/__main__.py` file also supports package-style execution when the repository root is on `PYTHONPATH`.

## SUMO-specific notes

- `src/simulation/traci_interface.py` depends on `SUMO_HOME` being set or SUMO binaries being available on `PATH`.
- SUMO model files and generated scenario artifacts live under `sumo/`, `simulations/`, and `data/`.
- Avoid changing generated SUMO network/route XML manually unless the task is explicitly about scenario creation.

## Useful links

- [README.md](README.md)
- [SUMO_ENV_SETUP.md](SUMO_ENV_SETUP.md)
- [README_SIMULATION.md](README_SIMULATION.md)
- [pyproject.toml](pyproject.toml)
- [requirements.txt](requirements.txt)

## Advice for code changes

- Preserve the domain semantics of EV battery state, travel times, station occupancy, and charging schedules.
- Keep the simulation and environment responsibilities separate: `src/simulation/` for SUMO/TraCI, `src/rl_env/` for training/reward mechanics.
- When editing tests, prefer adding focused regression coverage in `tests/` rather than broad end-to-end changes.
