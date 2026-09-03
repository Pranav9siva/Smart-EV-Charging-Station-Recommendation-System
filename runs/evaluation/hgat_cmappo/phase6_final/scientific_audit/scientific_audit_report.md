# Phase 6 Scientific Validity Audit Report

**Audit Target**: `runs/evaluation/hgat_cmappo/phase6_final/`  
**Total Completed SUMO Runs Audited**: 450 (5 Models $\times$ 90 Scenario Configs)  
**Overall Scientific Audit Status**: **VALID_WITH_CAVEATS**

---

## 1. Categorized Validity Breakdown

### A. Execution / Accounting Validity: **VALID**
- 450 expected scenario runs strictly match the 450 completed episode records.
- 0 missing runs, 0 duplicate runs.
- 75 total 1000-EV SUMO simulation runs (15 episodes per model) completed cleanly.

### B. Metric Provenance Validity: **VALID**
- All per-episode metrics are 100% mathematically reconstructable from raw logged events (`charging_events.csv`, `queue_events.csv`, `sumo_events.csv`, `constraint_events.csv`).
- Reference route distance ($5.0\text{ km}$) and additional detour distance ($1.15\text{ km}$ net station branch) are cleanly separated.

### C. Statistical Validity: **VALID_WITH_CAVEATS**
- **Detour Distance**: Statistically significant difference demonstrated ($p < 0.0001$, Cohen's $d_z = -3.25$ vs PPO, $-6.88$ vs MAPPO).
- **Waiting Time & Cost**: Low per-step variance across seeds under 100 EV demand settings.

### D. Claim Validity: **VALID_WITH_CAVEATS**
- **Detour Reduction**: Fully validated by raw event data.
- **Constraint Safety**: Validated by state predicate logs.
- **Scalability**: 1000-EV runs completed; queue waiting differentiation requires higher spatial crowding stress.

---

## 2. Final Audit Status Verdict

**FINAL SCIENTIFIC AUDIT STATUS**: **VALID_WITH_CAVEATS**
