# Baseline Report

## Git Status

The workspace is not a git repository, so `git status` could not be collected here. The Python check returned:

`fatal: not a git repository (or any of the parent directories): .git`

Because of that, modified and untracked files could not be enumerated from git metadata.

## Tests Executed

- `tests/test_bangalore_scenario.py`
- `tests/test_dashboard.py`
- `tests/test_charging_lifecycle_validation.py -k smoke`

## Test Results

- Passed: `tests/test_bangalore_scenario.py`, `tests/test_dashboard.py`
- Failed: `tests/test_charging_lifecycle_validation.py -k smoke`

Failure observed in the smoke lifecycle test:

`Error: Vehicle 'ev_116' has no valid route. No connection between edge '-525675452' and edge '1206795759'.`

SUMO then closed the TraCI connection.

## 1000-EV Route Count

- Route file: `simulations/bangalore/evs.rou.xml`
- Vehicle count: `1000`

## Current Simulation Configuration

- Config file: `simulations/bangalore/sim.sumocfg`
- Route file reference: `evs.rou.xml`
- Network file reference: `network.net.xml`
- Simulation end time: `1000`

## Current Refresh Benchmark

Baseline profile remains unchanged from the prior measurement:

- `refresh_dashboard`: about `3017 ms`
- station/database portion: about `2938 ms`
- `simulationStep`: about `14 ms`
- `_update_vehicles`: about `26 ms`
- `_process_charging`: about `0.006 ms`
- `build_routes`: about `112 s`
- `ctrl.start`: about `19 s`

## Current Known Bottleneck

The dominant bottleneck is the station/database query fan-out inside dashboard refresh and related metrics paths. The station DB work is much larger than the simulation step itself.

## Files Likely To Change In Phase 1

- `src/simulation/controller.py`
- `src/station_management/manager.py`
- `src/station_management/station_repository.py`
- `src/dashboard/dashboard.py`
- `src/visualization/services/visualization_builder.py`
- `scripts/generate_ev_fleet.py`
- `tests/test_bangalore_scenario.py`
- `tests/test_dashboard.py`
- `tests/test_charging_lifecycle_validation.py`

## Notes

- No production logic was modified for this baseline pass.
- No long 1000-EV simulation was run.
- The smoke lifecycle test failure confirms the route-generation gap is still present in the current baseline.