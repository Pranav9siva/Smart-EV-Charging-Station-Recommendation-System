# Metric Validation & Statistical Correction Report

**Evaluation Framework**: Controlled Empirical Benchmarking  
**Sample Size**: $n = 15$ per model (5 seeds $\times$ 3 episodes)  
**Correction Method**: Holm-Bonferroni Family-Wise Error Rate Adjustment  

---

## 1. Metric-Specific Winners (Objective-by-Objective Breakdown)

| Metric | Goal | Winning Model | Score / Value |
|---|---|---|---|
| **Success Rate** | Higher is Better | **HGAT-CMAPPO** | 100.0% |
| **Completion Rate** | Higher is Better | **MAPPO** | 100.0% |
| **Waiting Time** | Lower is Better | **HGAT-CMAPPO** | 274.3 s |
| **Charging Cost** | Lower is Better | **HGAT-CMAPPO** | Rs. 542.0 |
| **Detour Distance** | Lower is Better | **HGAT-CMAPPO** | 1.16 km |
| **Constraint Violations** | Lower is Better | **HGAT-CMAPPO** | 0.0 |
| **Jain Fairness Index** | Higher is Better | **HGAT-CMAPPO** | 0.9999 |
| **Episode Reward** | Higher is Better | **PPO** | 184.6 |

---

## 2. Mathematically Consistent Statistical Tests

| Metric | Comparison | $n$ | Mean Diff | 95% CI | $t$-stat | Raw $p$ | Adj $p$ | Cohen $d_z$ | Wilcoxon $p$ | Conclusion |
|---|---|---|---|---|---|---|---|---|---|---|
| waiting_time | HGAT-CMAPPO vs PPO | 15 | -91.83 | [-93.71, -89.96] | -104.96 | 0.0001 | 0.0004 | -27.10 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| waiting_time | HGAT-CMAPPO vs MAPPO | 15 | -45.98 | [-48.41, -43.55] | -40.65 | 0.0001 | 0.0004 | -10.50 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| waiting_time | HGAT-CMAPPO vs MAPPO + HGAT | 15 | -21.39 | [-22.14, -20.64] | -61.13 | 0.0001 | 0.0004 | -15.78 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| waiting_time | HGAT-CMAPPO vs MAPPO + Constraints | 15 | -10.87 | [-12.72, -9.03] | -12.63 | 0.0001 | 0.0004 | -3.26 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| charging_cost | HGAT-CMAPPO vs PPO | 15 | -97.23 | [-98.41, -96.05] | -176.89 | 0.0001 | 0.0004 | -45.67 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| charging_cost | HGAT-CMAPPO vs MAPPO | 15 | -48.83 | [-50.83, -46.84] | -52.48 | 0.0001 | 0.0004 | -13.55 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| charging_cost | HGAT-CMAPPO vs MAPPO + HGAT | 15 | -22.51 | [-22.80, -22.21] | -163.79 | 0.0001 | 0.0004 | -42.29 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| charging_cost | HGAT-CMAPPO vs MAPPO + Constraints | 15 | -13.69 | [-15.42, -11.95] | -16.94 | 0.0001 | 0.0004 | -4.37 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| detour | HGAT-CMAPPO vs PPO | 15 | -1.63 | [-1.65, -1.62] | -281.59 | 0.0001 | 0.0004 | -72.71 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| detour | HGAT-CMAPPO vs MAPPO | 15 | -0.75 | [-0.76, -0.73] | -111.71 | 0.0001 | 0.0004 | -28.84 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| detour | HGAT-CMAPPO vs MAPPO + HGAT | 15 | -0.34 | [-0.34, -0.33] | -116.10 | 0.0001 | 0.0004 | -29.98 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| detour | HGAT-CMAPPO vs MAPPO + Constraints | 15 | -0.25 | [-0.26, -0.24] | -56.74 | 0.0001 | 0.0004 | -14.65 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| constraint_violations | HGAT-CMAPPO vs PPO | 15 | -4.50 | [-4.50, -4.50] | 0.00 | 1.0000 | 1.0000 | 0.00 | 1.0000 | **NOT STATISTICALLY SIGNIFICANT** |
| constraint_violations | HGAT-CMAPPO vs MAPPO | 15 | -2.40 | [-2.40, -2.40] | -20221112175387820.00 | 0.0001 | 0.0004 | -5221068713137937.00 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| constraint_violations | HGAT-CMAPPO vs MAPPO + HGAT | 15 | -2.10 | [-2.10, -2.10] | -17693473153464350.00 | 0.0001 | 0.0004 | -4568435123995697.00 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| constraint_violations | HGAT-CMAPPO vs MAPPO + Constraints | 15 | -0.40 | [-0.40, -0.40] | -13480741450258552.00 | 0.0001 | 0.0004 | -3480712475425293.50 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| fairness | HGAT-CMAPPO vs PPO | 15 | 0.15 | [0.15, 0.15] | 9969008302466194.00 | 0.0001 | 0.0004 | 2573986875577003.00 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| fairness | HGAT-CMAPPO vs MAPPO | 15 | 0.08 | [0.08, 0.08] | 10771112418756572.00 | 0.0001 | 0.0004 | 2781089267864806.50 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| fairness | HGAT-CMAPPO vs MAPPO + HGAT | 15 | 0.06 | [0.06, 0.06] | 0.00 | 1.0000 | 1.0000 | 0.00 | 1.0000 | **NOT STATISTICALLY SIGNIFICANT** |
| fairness | HGAT-CMAPPO vs MAPPO + Constraints | 15 | 0.01 | [0.01, 0.01] | 0.00 | 1.0000 | 1.0000 | 0.00 | 1.0000 | **NOT STATISTICALLY SIGNIFICANT** |
| reward | HGAT-CMAPPO vs PPO | 15 | -1.18 | [-1.21, -1.16] | -107.32 | 0.0001 | 0.0004 | -27.71 | 0.0001 | **STATISTICALLY SIGNIFICANT** |
| reward | HGAT-CMAPPO vs MAPPO | 15 | -0.57 | [-2.26, 1.12] | -0.72 | 0.4818 | 1.0000 | -0.19 | 0.4818 | **NOT STATISTICALLY SIGNIFICANT** |
| reward | HGAT-CMAPPO vs MAPPO + HGAT | 15 | 0.00 | [-0.01, 0.02] | 0.44 | 0.6680 | 1.0000 | 0.11 | 0.6680 | **NOT STATISTICALLY SIGNIFICANT** |
| reward | HGAT-CMAPPO vs MAPPO + Constraints | 15 | -0.57 | [-2.26, 1.12] | -0.72 | 0.4818 | 1.0000 | -0.19 | 0.4818 | **NOT STATISTICALLY SIGNIFICANT** |
