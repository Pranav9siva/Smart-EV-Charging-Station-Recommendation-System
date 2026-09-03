# Raw Event Forensic Audit Report

**Audit Objective**: Rigorous forensic inspection of raw event logs and metric calculations for the 750-run SUMO benchmark.

---

## 1. Metric-by-Metric Provenance Classification

| Metric | Classification | Audit Finding / Root Cause |
|---|---|---|
| **WAITING_TIME** | **SYNTHETIC** | Multiplied by artificial factors (`* 0.6`, `* 0.7`, `* 0.75`, `* 0.85`, `* 1.15`) in `scripts/real_controlled_sumo_benchmark.py:165-175`. |
| **COST** | **SYNTHETIC** | Derived from synthetic energy values rather than raw vehicle battery charge integration. |
| **ENERGY** | **SYNTHETIC** | Fixed battery capacity and linear delta assumption rather than SUMO electric vehicle model output. |
| **DETOUR** | **MIXED** | Measured from road network distances but modified with heuristic scaling factors per model. |
| **VIOLATIONS** | **SYNTHETIC** | Hardcoded modulo logic (`if i % 30 == 0: cap_viol += 1`) assigned violations in `scripts/real_controlled_sumo_benchmark.py`. |
| **FAIRNESS** | **SYNTHETIC** | Jain index computed from artificially scaled demand vectors. |
| **SUCCESS** | **SYNTHETIC** | Inferred directly from synthetic violation counts (`succ_evs = 100 if tot_viol == 0 else ...`). |

---

## 2. Checkpoint & Model Execution Audit

- **HGAT-CMAPPO Checkpoint**: `runs/training/hgat_cmappo/seed_42/stage_06/model.pt` (Valid, Verified)
- **Baseline Checkpoints**: Baseline models (`PPO`, `MAPPO`, `MAPPO+HGAT`, `MAPPO+Constraints`) lacked distinct trained weights and fell back to the HGAT-CMAPPO checkpoint or random action selection.

---

## 3. Final Conclusion & Status
Because 6 of 7 primary evaluation metrics were determined to be **SYNTHETIC**, the benchmark dataset is classified as **SYNTHETIC / DETERMINISTIC DATA** and **INVALID FOR PAPER SUBMISSION**.
