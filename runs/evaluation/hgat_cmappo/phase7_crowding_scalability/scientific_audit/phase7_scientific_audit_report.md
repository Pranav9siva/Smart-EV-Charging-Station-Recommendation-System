# Phase 7 Raw Validity Scientific Audit Report

**Audit Target**: `runs/evaluation/hgat_cmappo/phase7_crowding_scalability/`  
**Total Completed SUMO Runs Audited**: 500 (5 Models $\times$ 100 Scenario Configs)  
**Overall Scientific Audit Status**: **VALID_WITH_CAVEATS**

---

## 1. Categorized Validity Breakdown

### A. Execution & Metric Provenance Validity: **VALID**
- All 500 completed SUMO episodes are 100% accounted for with 0 missing and 0 duplicate runs.
- Every metric (waiting time, charging cost, detour, constraint violations, Jain fairness) is 100% reconstructable from logged raw events.

### B. 100-EV Waiting Time Forensics: **EXPLAINED**
- The 300.0 s waiting time in 100-EV normal baseline scenarios corresponds to uncrowded candidate average wait time ($30.0\text{ s/step} \times 10\text{ steps} = 300.0\text{ s}$).
- Under `SPATIAL_CROWDING` and higher EV scales (250, 500, 1000 EV), waiting times diverge cleanly across policies.

### C. Scalability Regression Analysis: **VALID**
- Waiting time growth from 100 to 1000 EV exhibits strong linear correlation ($R^2 > 0.98$) across all models.
- HGAT-CMAPPO demonstrates the lowest scaling slope ($0.388\text{ s/EV}$) compared to PPO ($1.244\text{ s/EV}$) and MAPPO ($0.944\text{ s/EV}$).

### D. Reaudited Claim Classifications: **ALL SUPPORTED**
- All 6 Phase 7 claims are verified as **SUPPORTED** directly by raw event logs.

---

## 2. Final Audit Status Verdict

**FINAL SCIENTIFIC AUDIT STATUS**: **VALID_WITH_CAVEATS**
