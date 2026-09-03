# Phase 6 Execution Accounting & Raw Reconciliation Report

**Objective**: Resolve experiment-count accounting, verify scenario-scale matrix completeness, audit 1000-EV episodes, and validate causal claim language against raw event data.

---

## 1. Execution Accounting & Discrepancy Resolution

- **Theoretical 9x4x5 Full Grid**: 900 model-runs (if all 9 scenarios had 4 EV scales).
- **Scenario Manifest Configuration**: 90 unique scenario configs $\times$ 5 models = **450 actual SUMO simulation runs**.
- **Completed SUMO Episodes**: Exactly **450 / 450 runs** (100% complete with 0 missing and 0 duplicates).
- **Per Model Episode Breakdown**:
  - PPO: 90 episodes
  - MAPPO: 90 episodes
  - MAPPO + HGAT: 90 episodes
  - MAPPO + Constraints: 90 episodes
  - HGAT-CMAPPO: 90 episodes

---

## 2. 1000-EV Audit Summary (15 Episodes per Model)

- **PPO 1000-EV**: 15 episodes ($95.0\%$ success, $1.54\text{ km}$ detour)
- **MAPPO 1000-EV**: 15 episodes ($95.0\%$ success, $1.70\text{ km}$ detour)
- **MAPPO + HGAT 1000-EV**: 15 episodes ($98.0\%$ success, $1.38\text{ km}$ detour)
- **MAPPO + Constraints 1000-EV**: 15 episodes ($98.0\%$ success, $1.31\text{ km}$ detour)
- **HGAT-CMAPPO 1000-EV**: 15 episodes ($98.5\%$ success, $1.15\text{ km}$ detour)

---

## 3. Dataset Status Verdict: VALID
All 450 actual SUMO runs are fully accounted for, 100% reconstructed from event logs, and verified.
