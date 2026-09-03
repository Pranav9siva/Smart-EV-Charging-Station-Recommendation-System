"""SUMO Integration Test for HGAT-CMAPPO Model."""

from __future__ import annotations

import csv
from pathlib import Path
import pytest
import torch

from src.rl_env.standardized_ev_env import StandardizedEVEnv
from src.rl.hgat.graph_builder import HeteroGraphBuilder
from src.rl.mappo.actor_critic import HGAT_CMAPPO_Model


def test_hgat_cmappo_sumo_integration_trace() -> None:
    env = StandardizedEVEnv(num_evs=10, seed=42)
    obs, info = env.reset()

    model = HGAT_CMAPPO_Model(embedding_dim=64, candidate_count=8)
    builder = HeteroGraphBuilder(candidate_count=8)

    # 1. Build Graph from environment step state
    tracked = env.vehicle_manager.list_tracked_vehicles()
    ev_states = []
    for i, vrec in enumerate(tracked):
        lat, lon = env.ev_positions[i]
        ev_states.append({
            "ev_id": vrec.vehicle_id,
            "battery_pct": vrec.battery_pct,
            "battery_capacity_kwh": vrec.battery_capacity_kwh,
            "remaining_range_km": vrec.remaining_range_km,
            "lat": lat,
            "lon": lon,
            "dest_lat": 12.98,
            "dest_lon": 77.60,
            "speed": 10.0,
            "step": env.step_count,
        })
    candidate_stations = env.current_candidates

    graph = builder.build_graph(ev_states, candidate_stations)
    cand_indices = [[j for j in range(len(cand))] for cand in candidate_stations]

    # 2. Forward pass through HGAT-CMAPPO
    actions, log_probs, values, dist = model(graph, cand_indices)
    actions_np = actions.detach().cpu().numpy()

    # 3. Environment Step
    next_obs, rewards, dones, truncated, step_info = env.step(actions_np)

    # 4. Save Integration Trace CSV
    trace_dir = Path("runs/evaluation/hgat_cmappo")
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace_dir / "integration_trace.csv"

    with open(trace_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "step",
            "ev_id",
            "soc",
            "selected_action",
            "selected_station_id",
            "travel_distance_km",
            "waiting_time_min",
            "charging_cost",
            "reward",
        ])

        for i, ev_s in enumerate(ev_states):
            act_idx = int(actions_np[i])
            cands = candidate_stations[i]
            chosen_sid = cands[act_idx]["station_id"] if act_idx < len(cands) else "none"
            dist_km = cands[act_idx]["distance_km"] if act_idx < len(cands) else 0.0
            wait_min = cands[act_idx]["avg_wait_min"] if act_idx < len(cands) else 0.0
            cost = cands[act_idx]["price_per_kwh"] * 20.0 if act_idx < len(cands) else 0.0

            rew_val = float(rewards) if isinstance(rewards, (int, float)) else float(rewards[i])
            writer.writerow([
                0,
                1,
                ev_s.get("ev_id", f"ev_{i}"),
                f"{float(ev_s.get('battery_pct', 50.0)):.1f}",
                act_idx,
                chosen_sid,
                f"{dist_km:.2f}",
                f"{wait_min:.1f}",
                f"{cost:.2f}",
                f"{rew_val:.2f}",
            ])

    assert trace_file.exists()
    assert trace_file.stat().st_size > 0
    assert rewards is not None
