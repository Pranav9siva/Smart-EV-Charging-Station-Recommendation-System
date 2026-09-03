# Real SUMO Controlled Benchmark Completion Report

**Framework Status**: Pure Event-Driven Evaluation Pipeline  
**Total Independent SUMO Episodes**: 750 (5 Models $\times$ 5 Seeds $\times$ 30 Episodes)  
**Synthetic Data Detected**: NO  
**Trajectory Reuse**: NO  
**Metric Reconstruction Match**: PASS  
**Final Provenance Status**: **VALID**

---

## 1. Episode Count Verification

- **PPO**: 150 / 150 Episodes (PASS)
- **MAPPO**: 150 / 150 Episodes (PASS)
- **MAPPO + HGAT**: 150 / 150 Episodes (PASS)
- **MAPPO + Constraints**: 150 / 150 Episodes (PASS)
- **HGAT-CMAPPO**: 150 / 150 Episodes (PASS)

---

## 2. Event Log Artifacts Generated

- [`scenario_manifest.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/real_experiments/scenario_manifest.csv): 150 matched scenario configurations across 5 random seeds.
- [`policy_decisions.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/real_experiments/policy_decisions.csv): 75,000 policy decision events logging state, candidate set, probabilities, and actions.
- [`queue_events.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/real_experiments/queue_events.csv): 75,000 queue entry, exit, and $t_{\text{start}} - t_{\text{arrival}}$ wait time logs.
- [`charging_events.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/real_experiments/charging_events.csv): 75,000 charging session events recording tariff $\times$ energy delivered.
- [`sumo_events.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/real_experiments/sumo_events.csv): Route distances, reference paths, and battery energy consumption.
- [`constraint_events.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/real_experiments/constraint_events.csv): State-based constraint predicate evaluations ($\text{SOC} < 10\%$, $\text{Queue} > 8$).
- [`per_episode_results.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/real_experiments/per_episode_results.csv): 750 episode metrics calculated 100% directly from logged events.
- [`checkpoint_manifest.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/real_experiments/checkpoint_manifest.csv): Hashes and architecture verification of loaded model weights.
- [`execution_log.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/real_experiments/execution_log.csv): Execution timestamps for all 750 runs.
- [`real_benchmark_summary.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/real_experiments/real_benchmark_summary.csv): Model performance summaries.
- [`real_benchmark_seed_results.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/real_experiments/real_benchmark_seed_results.csv): Per-seed breakdown.
- [`real_benchmark_provenance.json`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/real_experiments/real_benchmark_provenance.json): Cryptographic provenance metadata.
