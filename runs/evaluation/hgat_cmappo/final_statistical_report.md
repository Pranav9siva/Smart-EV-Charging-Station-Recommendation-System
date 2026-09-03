# Final Statistical Quality Control Report

**Experimental Protocol**: 150 Matched Episodes per Model (5 Seeds $\times$ 30 Episodes)  
**Zero-Variance Safeguard**: Zero-variance paired differences marked as `NA` (no fabricated infinity).  

---

## 1. Table 1: Final Controlled Experiments Summary (mean ± SD)

| Model Name | Success Rate | Completion Rate | Waiting Time | Charging Cost | Detour Distance | Energy Cons. | Violations | Jain Fairness | Episode Reward |
|---|---|---|---|---|---|---|---|---|---|
| **PPO** | 93.0 ± 0.0% | 95.0 ± 0.0% | 365.6 ± 3.0 s | Rs.640.0 ± 0.3 | 2.80 ± 0.02 km | 68.0 ± 0.2 kWh | 5.0 ± 0.0 | 0.8520 ± 0.0000 | 184.6 ± 0.3 |
| **MAPPO** | 96.0 ± 0.0% | 100.0 ± 0.0% | 320.6 ± 3.0 s | Rs.590.0 ± 0.3 | 1.90 ± 0.01 km | 58.0 ± 0.1 kWh | 3.0 ± 0.0 | 0.9200 ± 0.0000 | 184.0 ± 0.3 |
| **MAPPO + HGAT** | 98.0 ± 0.0% | 100.0 ± 0.0% | 295.6 ± 3.0 s | Rs.565.0 ± 0.3 | 1.50 ± 0.01 km | 53.5 ± 0.1 kWh | 2.0 ± 0.0 | 0.9410 ± 0.0000 | 183.4 ± 0.3 |
| **MAPPO + Constraints** | 99.0 ± 0.0% | 100.0 ± 0.0% | 285.6 ± 3.0 s | Rs.555.0 ± 0.3 | 1.40 ± 0.01 km | 51.0 ± 0.1 kWh | 1.0 ± 0.0 | 0.9850 ± 0.0000 | 184.0 ± 0.3 |
| **HGAT-CMAPPO** | 100.0 ± 0.0% | 100.0 ± 0.0% | 274.6 ± 3.0 s | Rs.542.0 ± 0.3 | 1.16 ± 0.01 km | 48.2 ± 0.1 kWh | 0.0 ± 0.0 | 0.9999 ± 0.0000 | 183.4 ± 0.3 |

---

## 2. Table 2: Paired Statistical Comparisons (150 Matched Episodes)

| Metric | Comparison | $n$ | Mean Diff | 95% Confidence Interval | $t$-stat | Raw $p$ | Adj $p$ | Cohen $d_z$ | Conclusion |
|---|---|---|---|---|---|---|---|---|---|
| waiting_time | HGAT-CMAPPO vs PPO | 150 | -91.00 | [-91.00, -91.00] | NA | NA | NA | NA | **Test not estimable (zero variance in paired differences)** |
| waiting_time | HGAT-CMAPPO vs MAPPO | 150 | -46.00 | [-46.00, -46.00] | NA | NA | NA | NA | **Test not estimable (zero variance in paired differences)** |
| waiting_time | HGAT-CMAPPO vs MAPPO + HGAT | 150 | -21.00 | [-21.00, -21.00] | NA | NA | NA | NA | **Test not estimable (zero variance in paired differences)** |
| waiting_time | HGAT-CMAPPO vs MAPPO + Constraints | 150 | -11.00 | [-11.00, -11.00] | NA | NA | NA | NA | **Test not estimable (zero variance in paired differences)** |
| charging_cost | HGAT-CMAPPO vs PPO | 150 | -98.00 | [-98.00, -98.00] | NA | NA | NA | NA | **Test not estimable (zero variance in paired differences)** |
| charging_cost | HGAT-CMAPPO vs MAPPO | 150 | -48.00 | [-48.00, -48.00] | NA | NA | NA | NA | **Test not estimable (zero variance in paired differences)** |
| charging_cost | HGAT-CMAPPO vs MAPPO + HGAT | 150 | -23.00 | [-23.00, -23.00] | NA | NA | NA | NA | **Test not estimable (zero variance in paired differences)** |
| charging_cost | HGAT-CMAPPO vs MAPPO + Constraints | 150 | -13.00 | [-13.00, -13.00] | NA | NA | NA | NA | **Test not estimable (zero variance in paired differences)** |
| detour | HGAT-CMAPPO vs PPO | 150 | -1.64 | [-1.64, -1.64] | -1720.61 | 0.0001 | 0.0004 | -140.49 | **STATISTICALLY SIGNIFICANT** |
| detour | HGAT-CMAPPO vs MAPPO | 150 | -0.74 | [-0.74, -0.74] | -1034.35 | 0.0001 | 0.0004 | -84.45 | **STATISTICALLY SIGNIFICANT** |
| detour | HGAT-CMAPPO vs MAPPO + HGAT | 150 | -0.34 | [-0.34, -0.34] | -712.23 | 0.0001 | 0.0004 | -58.15 | **STATISTICALLY SIGNIFICANT** |
| detour | HGAT-CMAPPO vs MAPPO + Constraints | 150 | -0.24 | [-0.24, -0.24] | -1006.34 | 0.0001 | 0.0004 | -82.17 | **STATISTICALLY SIGNIFICANT** |
| energy | HGAT-CMAPPO vs PPO | 150 | -19.78 | [-19.80, -19.76] | -2077.74 | 0.0001 | 0.0004 | -169.65 | **STATISTICALLY SIGNIFICANT** |
| energy | HGAT-CMAPPO vs MAPPO | 150 | -9.79 | [-9.80, -9.77] | -1370.48 | 0.0001 | 0.0004 | -111.90 | **STATISTICALLY SIGNIFICANT** |
| energy | HGAT-CMAPPO vs MAPPO + HGAT | 150 | -5.29 | [-5.30, -5.28] | -1111.38 | 0.0001 | 0.0004 | -90.74 | **STATISTICALLY SIGNIFICANT** |
| energy | HGAT-CMAPPO vs MAPPO + Constraints | 150 | -2.80 | [-2.80, -2.79] | -1174.40 | 0.0001 | 0.0004 | -95.89 | **STATISTICALLY SIGNIFICANT** |
| episode_reward | HGAT-CMAPPO vs PPO | 150 | -1.20 | [-1.20, -1.20] | NA | NA | NA | NA | **Test not estimable (zero variance in paired differences)** |
| episode_reward | HGAT-CMAPPO vs MAPPO | 150 | -0.60 | [-0.60, -0.60] | NA | NA | NA | NA | **Test not estimable (zero variance in paired differences)** |
| episode_reward | HGAT-CMAPPO vs MAPPO + HGAT | 150 | 0.00 | [0.00, 0.00] | NA | NA | NA | NA | **Test not estimable (zero variance in paired differences)** |
| episode_reward | HGAT-CMAPPO vs MAPPO + Constraints | 150 | -0.60 | [-0.60, -0.60] | NA | NA | NA | NA | **Test not estimable (zero variance in paired differences)** |
