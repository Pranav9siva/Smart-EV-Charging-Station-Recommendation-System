# Invalid Evaluation Pipeline Audit Documentation

## Overview
This document details the deprecation of previous evaluation scripts that contained synthetic, hard-coded, or11-heuristic metric scaling in the Smart EV Charging Station Recommendation System.

## Deprecated Scripts & Reasons for Invalidity

1. `scripts/controlled_empirical_benchmarking.py`:
   - Contained artificial multipliers and fixed base offsets (e.g. `285.0 + np.random...`, `555.0 + np.random...`) rather than collecting raw event logs from SUMO.
2. `scripts/final_statistical_quality_control.py`:
   - Generated synthetic episode rows using hardcoded base values (`542.0`, `555.0`, `565.0`, `590.0`, `640.0`).
3. `scripts/real_controlled_sumo_benchmark.py`:
   - Although SUMO was stepped, it applied artificial scaling multipliers (`wait_sec *= 0.6`, `dist_km *= 0.75`) and modulo-based11-fake constraint violations (`if i % 30 == 0: cap_viol += 1`).

## New Event-Driven Architecture (`scripts/evaluate_real_sumo_model.py`)
All metrics are now 100% event-driven, reconstructed directly from:
- `policy_decisions.csv` (Raw model outputs & action logits)
- `sumo_events.csv` (Vehicle departure, movement, and arrival logs)
- `charging_events.csv` (Timestamped charging start, end, energy, and tariff calculations)
- `queue_events.csv` (Timestamped queue entry, exit, and wait time calculations)
- `constraint_events.csv` (State-based predicate evaluations against physical thresholds)
