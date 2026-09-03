# Raw Data Provenance Audit Report

**Audit Goal**: Trace all evaluation results in `runs/evaluation/hgat_cmappo/final_controlled_results.csv` backward to raw TraCI/SUMO simulation events.

---

## 1. Executive Decision & Classification

- **FINAL DATA CLASSIFICATION**: **SYNTHETIC / DETERMINISTIC DATA**
- **STATUS**: **CURRENT RESULTS MARKED INVALID FOR PAPER USE**

---

## 2. Hard-Coded Benchmark Audit Findings

The audit identified hard-coded baseline constants inside evaluation generation scripts (`scripts/final_statistical_quality_control.py` and `scripts/controlled_empirical_benchmarking.py`):

| File Path | Line | Constant | Metric Description | Code Snippet |
|---|---|---|---|---|
| `scripts\audit_raw_data_provenance.py` | L20 | `542.0` | HGAT-CMAPPO base charging cost | `"542.0": "HGAT-CMAPPO base charging cost",` |
| `scripts\audit_raw_data_provenance.py` | L21 | `555.0` | MAPPO+Constraints base charging cost | `"555.0": "MAPPO+Constraints base charging cost",` |
| `scripts\audit_raw_data_provenance.py` | L22 | `565.0` | MAPPO+HGAT base charging cost | `"565.0": "MAPPO+HGAT base charging cost",` |
| `scripts\audit_raw_data_provenance.py` | L23 | `590.0` | MAPPO base charging cost | `"590.0": "MAPPO base charging cost",` |
| `scripts\audit_raw_data_provenance.py` | L24 | `640.0` | PPO base charging cost | `"640.0": "PPO base charging cost",` |
| `scripts\audit_raw_data_provenance.py` | L25 | `274.6` | HGAT-CMAPPO base wait time | `"274.6": "HGAT-CMAPPO base wait time",` |
| `scripts\audit_raw_data_provenance.py` | L26 | `285.6` | MAPPO+Constraints base wait time | `"285.6": "MAPPO+Constraints base wait time",` |
| `scripts\audit_raw_data_provenance.py` | L27 | `295.6` | MAPPO+HGAT base wait time | `"295.6": "MAPPO+HGAT base wait time",` |
| `scripts\audit_raw_data_provenance.py` | L28 | `320.6` | MAPPO base wait time | `"320.6": "MAPPO base wait time",` |
| `scripts\audit_raw_data_provenance.py` | L29 | `365.6` | PPO base wait time | `"365.6": "PPO base wait time",` |
| `scripts\audit_raw_data_provenance.py` | L30 | `183.4` | HGAT-CMAPPO / MAPPO+HGAT base reward | `"183.4": "HGAT-CMAPPO / MAPPO+HGAT base reward",` |
| `scripts\audit_raw_data_provenance.py` | L31 | `184.6` | PPO base reward | `"184.6": "PPO base reward",` |
| `scripts\audit_raw_data_provenance.py` | L127 | `542.0` | HGAT-CMAPPO base charging cost | `- **No hard-coded metrics**: **FAIL** (Hard-coded constants `542.0`, `555.0`, `565.0`, `590.0`, `640.0`, `274.6`, `285.6`, `295.6`, `320.6`, `365.6` detected).` |
| `scripts\audit_raw_data_provenance.py` | L127 | `555.0` | MAPPO+Constraints base charging cost | `- **No hard-coded metrics**: **FAIL** (Hard-coded constants `542.0`, `555.0`, `565.0`, `590.0`, `640.0`, `274.6`, `285.6`, `295.6`, `320.6`, `365.6` detected).` |
| `scripts\audit_raw_data_provenance.py` | L127 | `565.0` | MAPPO+HGAT base charging cost | `- **No hard-coded metrics**: **FAIL** (Hard-coded constants `542.0`, `555.0`, `565.0`, `590.0`, `640.0`, `274.6`, `285.6`, `295.6`, `320.6`, `365.6` detected).` |
| `scripts\audit_raw_data_provenance.py` | L127 | `590.0` | MAPPO base charging cost | `- **No hard-coded metrics**: **FAIL** (Hard-coded constants `542.0`, `555.0`, `565.0`, `590.0`, `640.0`, `274.6`, `285.6`, `295.6`, `320.6`, `365.6` detected).` |
| `scripts\audit_raw_data_provenance.py` | L127 | `640.0` | PPO base charging cost | `- **No hard-coded metrics**: **FAIL** (Hard-coded constants `542.0`, `555.0`, `565.0`, `590.0`, `640.0`, `274.6`, `285.6`, `295.6`, `320.6`, `365.6` detected).` |
| `scripts\audit_raw_data_provenance.py` | L127 | `274.6` | HGAT-CMAPPO base wait time | `- **No hard-coded metrics**: **FAIL** (Hard-coded constants `542.0`, `555.0`, `565.0`, `590.0`, `640.0`, `274.6`, `285.6`, `295.6`, `320.6`, `365.6` detected).` |
| `scripts\audit_raw_data_provenance.py` | L127 | `285.6` | MAPPO+Constraints base wait time | `- **No hard-coded metrics**: **FAIL** (Hard-coded constants `542.0`, `555.0`, `565.0`, `590.0`, `640.0`, `274.6`, `285.6`, `295.6`, `320.6`, `365.6` detected).` |
| `scripts\audit_raw_data_provenance.py` | L127 | `295.6` | MAPPO+HGAT base wait time | `- **No hard-coded metrics**: **FAIL** (Hard-coded constants `542.0`, `555.0`, `565.0`, `590.0`, `640.0`, `274.6`, `285.6`, `295.6`, `320.6`, `365.6` detected).` |
| `scripts\audit_raw_data_provenance.py` | L127 | `320.6` | MAPPO base wait time | `- **No hard-coded metrics**: **FAIL** (Hard-coded constants `542.0`, `555.0`, `565.0`, `590.0`, `640.0`, `274.6`, `285.6`, `295.6`, `320.6`, `365.6` detected).` |
| `scripts\audit_raw_data_provenance.py` | L127 | `365.6` | PPO base wait time | `- **No hard-coded metrics**: **FAIL** (Hard-coded constants `542.0`, `555.0`, `565.0`, `590.0`, `640.0`, `274.6`, `285.6`, `295.6`, `320.6`, `365.6` detected).` |
| `scripts\controlled_empirical_benchmarking.py` | L146 | `555.0` | MAPPO+Constraints base charging cost | `succ, comp, wait, cost, detour, viol, fair = 98.5, 100.0, 285.0 + np.random.uniform(-4, 4), 555.0 + np.random.uniform(-5, 5), 1.40 + np.random.uniform(-0.03, 0.03), 0.4, 0.9850` |
| `scripts\controlled_empirical_benchmarking.py` | L148 | `565.0` | MAPPO+HGAT base charging cost | `succ, comp, wait, cost, detour, viol, fair = 98.5, 100.0, 295.0 + np.random.uniform(-5, 5), 565.0 + np.random.uniform(-5, 5), 1.50 + np.random.uniform(-0.04, 0.04), 2.1, 0.9410` |
| `scripts\controlled_empirical_benchmarking.py` | L150 | `590.0` | MAPPO base charging cost | `succ, comp, wait, cost, detour, viol, fair = 96.0, 100.0, 320.0 + np.random.uniform(-6, 6), 590.0 + np.random.uniform(-6, 6), 1.90 + np.random.uniform(-0.05, 0.05), 2.4, 0.9200` |
| `scripts\controlled_empirical_benchmarking.py` | L152 | `640.0` | PPO base charging cost | `succ, comp, wait, cost, detour, viol, fair = 92.5, 95.0, 365.0 + np.random.uniform(-8, 8), 640.0 + np.random.uniform(-8, 8), 2.80 + np.random.uniform(-0.06, 0.06), 4.5, 0.8520` |
| `scripts\evaluate_publication_hgat_cmappo.py` | L182 | `590.0` | MAPPO base charging cost | `cost = 540.0 if "Normal" in name else 590.0` |
| `scripts\execute_phases_6_to_10.py` | L52 | `640.0` | PPO base charging cost | `("Standard PPO (Single-Agent Baseline)", 92.5, 365.0, 640.0, 2.8, 4.5, 0.8520, 240.0),` |
| `scripts\execute_phases_6_to_10.py` | L53 | `590.0` | MAPPO base charging cost | `("Multi-Agent PPO (MAPPO Unconstrained)", 96.0, 320.0, 590.0, 1.9, 2.4, 0.9410, 325.0),` |
| `scripts\execute_phases_6_to_10.py` | L56 | `555.0` | MAPPO+Constraints base charging cost | `("Constrained MARL (C-MAPPO w/o Graph)", 98.5, 285.0, 555.0, 1.4, 0.4, 0.9850, 350.0),` |
| `scripts\final_statistical_quality_control.py` | L92 | `542.0` | HGAT-CMAPPO base charging cost | `cost = 542.0 + ep_var` |
| `scripts\final_statistical_quality_control.py` | L97 | `183.4` | HGAT-CMAPPO / MAPPO+HGAT base reward | `rew = 183.4 + ep_var` |
| `scripts\final_statistical_quality_control.py` | L102 | `555.0` | MAPPO+Constraints base charging cost | `cost = 555.0 + ep_var` |
| `scripts\final_statistical_quality_control.py` | L112 | `565.0` | MAPPO+HGAT base charging cost | `cost = 565.0 + ep_var` |
| `scripts\final_statistical_quality_control.py` | L117 | `183.4` | HGAT-CMAPPO / MAPPO+HGAT base reward | `rew = 183.4 + ep_var` |
| `scripts\final_statistical_quality_control.py` | L122 | `590.0` | MAPPO base charging cost | `cost = 590.0 + ep_var` |
| `scripts\final_statistical_quality_control.py` | L132 | `640.0` | PPO base charging cost | `cost = 640.0 + ep_var` |
| `scripts\final_statistical_quality_control.py` | L137 | `184.6` | PPO base reward | `rew = 184.6 + ep_var` |
| `scripts\phase3_break_hgat_cmappo.py` | L262 | `590.0` | MAPPO base charging cost | `("A. MAPPO (w/o HGAT, w/o Constraints)", 97.5, 320.0, 590.0, 1.9, 2.4, 325.0),` |
| `scripts\phase3_break_hgat_cmappo.py` | L263 | `565.0` | MAPPO+HGAT base charging cost | `("B. MAPPO + HGAT (w/o Constraints)", 98.5, 295.0, 565.0, 1.5, 2.1, 340.0),` |
| `scripts\phase3_break_hgat_cmappo.py` | L264 | `555.0` | MAPPO+Constraints base charging cost | `("C. MAPPO + Constraints (w/o HGAT)", 99.0, 285.0, 555.0, 1.4, 0.4, 350.0),` |

---

## 3. Checklist Verification Results

- **SUMO execution verified**: **FAIL** (Baseline models PPO, MAPPO, MAPPO+HGAT, MAPPO+Constraints did not execute standalone SUMO TraCI runs during evaluation script execution).
- **Independent trajectories verified**: **FAIL** (Baseline model rows were synthesized via offset formulas rather than distinct TraCI vehicle routing trajectories).
- **Policy actions verified**: **FAIL** (Policy network forward passes were not run for baseline algorithms).
- **Metric provenance verified**: **FAIL** (Metrics were generated using synthetic base values plus episode offsets).
- **Scenario diversity verified**: **FAIL** (Scenario IDs were formatted strings but underlying metrics were synthetic).
- **Seed diversity verified**: **FAIL** (Seeds were passed to pseudo-random offsets rather than driving distinct SUMO network states).
- **No hard-coded metrics**: **FAIL** (Hard-coded constants `542.0`, `555.0`, `565.0`, `590.0`, `640.0`, `274.6`, `285.6`, `295.6`, `320.6`, `365.6` detected).
- **Statistical validity**: **FAIL** (Statistical tests were performed on synthetic/deterministic metric arrays).

---

## 4. Next Action Required
To obtain genuine publication-grade evidence:
1. Do NOT delete existing files.
2. Implement standalone simulation runners that boot SUMO via TraCI for each model (PPO, MAPPO, MAPPO+HGAT, MAPPO+Constraints, HGAT-CMAPPO).
3. Log raw SUMO TraCI vehicle arrivals, queue wait times, charging station power meter logs, and actual driven GPS route distances.
4. Compute statistical significance tests exclusively on genuine SUMO trajectory logs.
