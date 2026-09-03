# EV Digital Twin Dashboard

## Overview

The **EV Digital Twin Dashboard** is a real-time research visualization control center for the Smart EV Charging Station Recommendation System. It combines live SUMO/TraCI telemetry with frozen empirical research evidence from 1,450 experiment runs across Phases 6, 7, and 8.

---

## Dashboard Architecture & Distinction

| Visual Section | Data Source | Update Frequency | Purpose |
| :--- | :--- | :--- | :--- |
| **LIVE SIMULATION TELEMETRY** | SUMO / TraCI Trajectory (`dashboard_state.json`) | Real-time Polling (500ms) | Displays live waiting time, congestion, cumulative energy delivered, queue events, and port utilization for current SUMO simulation step ($0 \rightarrow 7200$). |
| **FROZEN RESEARCH EVIDENCE** | Paper-Safe Metrics (`outputs/performance_matrix.*`) | Static / Scope Filtered | Displays frozen publication benchmark evidence across 1,450 verified experiment episodes (500 Phase 7, 500 Phase 8, 450 Phase 6). |

---

## Canonical Entrypoints & File Mapping

1. **Primary Frontend Source**: [`outputs/dashboard.html`](file:///d:/Smart_EV_Station_Recommandadtion_System/outputs/dashboard.html)
2. **Synchronized Workspace Routes**:
   - `outputs/index.html`
   - `outputs/research_dashboard.html`
   - `dashboard/index.html`
   - `src/visualization/frontend/index.html`
   - `src/visualization/frontend/live_dashboard.html`

---

## Launching the Dashboard

### 1. Launch HTTP Server & SUMO Simulation
Execute `run_project.py` from the root directory:
```bash
python run_project.py
```
This starts the backend simulation loop and starts the HTTP state server on port `8765`.

### 2. Access the Dashboard
Open your browser and navigate to:
```
http://localhost:8765/dashboard.html
```

---

## State Generation Pipeline

`run_project.py` periodically computes step trajectory state and exports [`outputs/dashboard_state.json`](file:///d:/Smart_EV_Station_Recommandadtion_System/outputs/dashboard_state.json):
- `simulation`: `{ step, sim_time, congestion, simulation_status }`
- `kpis`: `{ avg_wait, cumulative_energy_delivered_kwh, charging_started, queue_events }`
- `step_history`: Full $0 \rightarrow 7200$ step trajectory buffer ensuring $100\%$ monotonic energy and KPI equality.
