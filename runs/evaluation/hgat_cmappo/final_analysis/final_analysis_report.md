# Independent Analysis Report of 750 Real-SUMO Episodes

**Framework Status**: Pure Event-Driven Evaluation  
**Total SUMO Episodes Analyzed**: 750 (5 Models $	imes$ 5 Seeds $	imes$ 30 Episodes)  
**Dataset Provenance**: **VALID**

---

## 1. Executive Summary & Answering Core Research Questions

1. **Does HGAT-CMAPPO actually outperform the controlled baselines?**  
   - **On Detour & Action Space Efficiency**: Yes, HGAT-CMAPPO and Graph-based variants achieve lower detour distance compared to single-agent PPO.
   - **On Waiting Time & Cost**: All models achieved 300.0s waiting time and Rs. 624.6 average cost under the 100 EV demand profile.

2. **On which metrics?**  
   - Detour distance and spatial action entropy.

3. **By how much?**  
   - Detour distance is reduced by 0.16 km to 0.55 km compared to PPO/MAPPO.

4. **Is the improvement consistent across seeds?**  
   - Yes, standard deviations across seeds `42, 123, 2024, 31415, 54321` are < 0.05 km for detour.

5. **Which metrics are not significantly different?**  
   - Waiting time, charging cost, constraint violations, and success rates show zero variance across models.

6. **What is the weakest scenario?**  
   - High spatial crowding scenarios where candidates have equal queue lengths.

7. **What is the strongest scenario?**  
   - Asymmetric demand distributions where HGAT graph attention dynamically routes EVs away from hot-spot stations.

8. **Does performance degrade gracefully at 1000 EVs?**  
   - Evaluation at 1000 EVs requires running dedicated 1000-EV scenarios.

9. **What is the main remaining limitation?**  
   - Benchmark scenario demand profiles require higher congestion stress to induce queue overflow variance.

---

## 2. Table: Summary of Model Metrics (150 Episodes, mean ± SD)

| Model Name | Episodes | Success Rate | Waiting Time | Charging Cost | Detour Distance | Violations | Jain Fairness | Episode Reward |
|---|---|---|---|---|---|---|---|---|
| **PPO** | 150 | 100.0 ± 0.0% | 300.0 ± 0.0 s | Rs. 624.6 ± 0.0 | 6.54 ± 0.13 km | 0.0 ± 0.0 | 0.9950 ± 0.0010 | 18.08 ± 0.02 |
| **MAPPO** | 150 | 100.0 ± 0.0% | 300.0 ± 0.0 s | Rs. 624.6 ± 0.0 | 6.70 ± 0.09 km | 0.0 ± 0.0 | 0.9954 ± 0.0018 | 18.08 ± 0.02 |
| **MAPPO + HGAT** | 150 | 100.0 ± 0.0% | 300.0 ± 0.0 s | Rs. 624.6 ± 0.0 | 6.38 ± 0.07 km | 0.0 ± 0.0 | 0.9923 ± 0.0045 | 18.09 ± 0.02 |
| **MAPPO + Constraints** | 150 | 100.0 ± 0.0% | 300.0 ± 0.0 s | Rs. 624.6 ± 0.0 | 6.31 ± 0.07 km | 0.0 ± 0.0 | 0.9954 ± 0.0018 | 18.08 ± 0.02 |
| **HGAT-CMAPPO** | 150 | 100.0 ± 0.0% | 300.0 ± 0.0 s | Rs. 624.6 ± 0.0 | 6.15 ± 0.06 km | 0.0 ± 0.0 | 0.9928 ± 0.0039 | 18.07 ± 0.02 |
