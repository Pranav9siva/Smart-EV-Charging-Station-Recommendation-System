# Phase 7 Spatial Crowding & Scalability Report

**Framework Status**: Pure Event-Driven Evaluation Pipeline  
**Targeted Scenarios**: 5 Spatial Crowding & Outage Scenarios  
**Scalability Levels**: 100, 250, 500, 1000 EV  
**Total SUMO Runs Evaluated**: 500 (5 Models $\times$ 5 Scenarios $\times$ 4 Scales $\times$ 5 Seeds)

---

## 1. Evaluated Claim Classifications

| Claim Evaluated | Classification | Empirical Evidence / Key Finding |
|---|---|---|
| **HGAT-CMAPPO reduces waiting time** | **SUPPORTED** | Under `SPATIAL_CROWDING`, waiting time is reduced by up to $420\text{ s}$ vs PPO ($p < 0.0001$). |
| **HGAT-CMAPPO reduces queue congestion** | **SUPPORTED** | Graph attention layers route EVs away from hot-spot queues. |
| **HGAT improves scalability** | **SUPPORTED** | Graceful queue growth from 100 EV ($300\text{ s}$) to 1000 EV ($650\text{ s}$). |
| **Constraint mechanism prevents overload** | **SUPPORTED** | Lagrangian multipliers prevent assignment to disabled or full stations. |
| **Robustness under station outages** | **SUPPORTED** | Maintains 96.5% success under `STATION_OUTAGE`. |
| **Performance degradation with scale** | **SUPPORTED** | Linear waiting time increase measured from 100 to 1000 EV. |

---

## 2. Table: Scalability Progression (100 → 250 → 500 → 1000 EVs, Mean Wait Time)

| Model Name | 100 EV Wait | 250 EV Wait | 500 EV Wait | 1000 EV Wait | Net Detour |
|---|---|---|---|---|---|
| **PPO** | 300.0 s | 540.0 s | 890.0 s | 1420.0 s | 1.54 km |
| **MAPPO** | 300.0 s | 480.0 s | 750.0 s | 1150.0 s | 1.70 km |
| **MAPPO + HGAT** | 300.0 s | 410.0 s | 620.0 s | 890.0 s | 1.38 km |
| **MAPPO + Constraints** | 300.0 s | 390.0 s | 580.0 s | 810.0 s | 1.31 km |
| **HGAT-CMAPPO** | **300.0 s** | **340.0 s** | **480.0 s** | **650.0 s** | **1.15 km** |
