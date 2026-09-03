# Changelog

All notable changes to the **Smart EV Charging Station Recommendation System** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v1.0.0] - 2026-09-03

### Added
- **HGAT-CMAPPO Architecture**: Implemented Heterogeneous Graph Attention Policy Networks (`src/rl/hgat/`) with multi-agent PPO coordination and adaptive Lagrangian constraint enforcers.
- **SUMO / TraCI Integration**: Microscopic traffic simulation integration (`src/simulation/`) on the Bengaluru metropolitan road network domain.
- **Phase 6 Accounting & Reconciliation**: 450 verified simulation runs ensuring exact state-event accounting across SUMO events, charging starts, and queue logs.
- **Phase 7 Demand Scaling Benchmark**: 500 verified simulation runs evaluating demand scaling up to 1,000 EVs and spatial station outage perturbations.
- **Phase 8 Component Ablation**: 500 verified simulation runs isolating structural contributions of coordination, graph attention embeddings, and Lagrangian safety guards.
- **EV Digital Twin Dashboard**: Real-time visualization control center featuring live SUMO telemetry (100% monotonic cumulative energy delivered, dynamic TraCI congestion, port utilization) alongside frozen research paper evidence.
- **Publication Evidence Package**: Paper-safe performance matrices (`outputs/performance_matrix.*`), claim registers, and audit reports (`runs/evaluation/hgat_cmappo/final_audit/`).
- **Reproducibility & Testing Suite**: `pytest` test suite and telemetry verification scripts.
- **Archive System**: Created `archive/invalid_evaluation_pipeline/` to preserve historical/invalid benchmark scripts for scientific provenance.

### Verified
- Zero telemetry mismatches between live chart trajectory points and state KPIs (`KPI === last history point`).
- Monotonic non-decreasing cumulative energy delivered ($E_t \ge E_{t-1}$).
- Full vertical page scrolling without component clipping across desktop viewports.
- Clean security audit with zero hardcoded API keys or credentials.
