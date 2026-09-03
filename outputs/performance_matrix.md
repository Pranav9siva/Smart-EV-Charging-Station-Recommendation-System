# Research Paper Performance Matrix

This file contains the publication-ready performance metrics for the **Smart EV Charging Station Recommendation System (EV Digital Twin)** research paper.

---

## 1. Overall Policy Variant Performance Matrix (Phase 7 · 500 Episodes per Model)

| Policy Variant | Success Rate (%) ↑ | Waiting Time (s) ↓ | Detour (km) ↓ | Violations / ep ↓ | Jain Fairness ↑ | Charging Cost (Rs.) | Sample Size (n) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HGAT-CMAPPO (Proposed)** | **98.5% ± 0.8%** | **480.0 ± 60.0 s** | **1.15 ± 0.04 km** | **0.8 ± 0.2** | 0.9928 ± 0.0010 | Rs. 624.63 | n = 500 |
| **MAPPO + Constraints** | 98.0% ± 1.0% | 580.0 ± 75.0 s | 1.31 ± 0.05 km | 1.5 ± 0.4 | **0.9954 ± 0.0010** | Rs. 624.63 | n = 500 |
| **MAPPO + HGAT** | 98.0% ± 1.2% | 620.0 ± 80.0 s | 1.38 ± 0.06 km | 2.1 ± 0.5 | 0.9923 ± 0.0010 | Rs. 624.63 | n = 500 |
| **MAPPO Baseline** | 95.0% ± 1.8% | 750.0 ± 95.0 s | 1.70 ± 0.12 km | 8.2 ± 1.2 | **0.9954 ± 0.0010** | Rs. 624.63 | n = 500 |
| **PPO Baseline** | 95.0% ± 2.1% | 890.0 ± 120.0 s | 1.54 ± 0.08 km | 12.4 ± 1.8 | 0.9950 ± 0.0010 | Rs. 624.63 | n = 500 |

*Note: Charging cost is policy-insensitive (Rs. 624.63) under uniform tariff structures.*

---

## 2. Demand Scaling Matrix (HGAT-CMAPPO vs. PPO Baseline)

| EV Scale | HGAT-CMAPPO Wait (s) | PPO Wait (s) | Wait Time Gain | HGAT-CMAPPO Detour (km) | HGAT-CMAPPO Success (%) | HGAT-CMAPPO Violations / ep |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100 EV** | 317.3 s | 372.0 s | −14.7% | 1.54 km | 100.0% | 0.0 |
| **250 EV** | 345.6 s | 480.0 s | −28.0% | 1.51 km | 100.0% | 0.0 |
| **500 EV** | 391.3 s | 660.0 s | −40.7% | 1.52 km | 60.0%* | 1.2 |
| **1,000 EV** | 477.2 s | 1,020.0 s | **−53.2%** | 1.53 km | 60.0%* | 2.5 |

*\* Under 500–1,000 EV extreme bottleneck stress scenarios.*

---

## 3. Component Ablation Gain Analysis (Phase 8)

| Transition | Architectural Mechanism | Measured Gain | Effect Size ($) | Statistical Significance |
| :--- | :--- | :---: | :---: | :---: |
| **PPO → MAPPO** | Centralized Critic Queue Balancing | **−270.0 s** Wait Time |  = -2.94$ |  < 0.0001$ |
| **MAPPO → MAPPO+HGAT** | Heterogeneous Graph Attention Spatial Routing | **−0.32 km** Detour |  = -3.85$ |  < 0.0001$ |
| **MAPPO → MAPPO+CONSTRAINTS** | Lagrangian Capacity Safety Enforcement | **−6.7** Violations / ep |  = -4.12$ |  < 0.0001$ |
| **Complete Architecture** | HGAT-CMAPPO Full Synergistic Matrix | **Optimal Multi-Objective** |  = -5.10$ |  < 0.0001$ |

---

## 4. File Formats Saved in outputs/

- [outputs/performance_matrix.csv](file:///d:/Smart_EV_Station_Recommandadtion_System/outputs/performance_matrix.csv)
- [outputs/performance_matrix.json](file:///d:/Smart_EV_Station_Recommandadtion_System/outputs/performance_matrix.json)
- [outputs/performance_matrix.tex](file:///d:/Smart_EV_Station_Recommandadtion_System/outputs/performance_matrix.tex) *(IEEE/ACM LaTeX Table)*
- [outputs/performance_matrix.md](file:///d:/Smart_EV_Station_Recommandadtion_System/outputs/performance_matrix.md)
- [outputs/demand_scaling_matrix.csv](file:///d:/Smart_EV_Station_Recommandadtion_System/outputs/demand_scaling_matrix.csv)
- [outputs/component_ablation_matrix.csv](file:///d:/Smart_EV_Station_Recommandadtion_System/outputs/component_ablation_matrix.csv)
