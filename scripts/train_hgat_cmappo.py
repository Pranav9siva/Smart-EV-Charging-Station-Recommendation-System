"""Curriculum Training Script for HGAT-CMAPPO Research Architecture."""

from __future__ import annotations

import json
import time
from pathlib import Path
import yaml
import torch
import torch.optim as optim
import numpy as np

from src.rl_env.standardized_ev_env import StandardizedEVEnv
from src.rl.hgat.graph_builder import HeteroGraphBuilder
from src.rl.mappo.actor_critic import HGAT_CMAPPO_Model
from src.rl.constraints.lagrangian import LagrangianConstraintSystem


def load_config(config_path: str = "configs/hgat_cmappo.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_curriculum():
    cfg = load_config()
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    curriculum_cfg = cfg.get("curriculum", {}).get("stages", [])

    seed = train_cfg.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    embedding_dim = model_cfg.get("embedding_dim", 64)
    candidate_count = train_cfg.get("candidate_count", 8)

    model = HGAT_CMAPPO_Model(embedding_dim=embedding_dim, candidate_count=candidate_count)
    lagrangian = LagrangianConstraintSystem(initial_lambda=cfg.get("constraints", {}).get("initial_lambda", 0.1))

    optimizer = optim.Adam([
        {"params": model.parameters(), "lr": train_cfg.get("actor_lr", 0.0003)},
        {"params": lagrangian.parameters(), "lr": train_cfg.get("lagrangian_lr", 0.01)},
    ])

    builder = HeteroGraphBuilder(candidate_count=candidate_count)
    base_save_dir = Path("runs/models/hgat_cmappo")
    base_save_dir.mkdir(parents=True, exist_ok=True)

    training_log_dir = Path("runs/training/hgat_cmappo")
    training_log_dir.mkdir(parents=True, exist_ok=True)

    history = []

    print("============================================================")
    print("STARTING HGAT-CMAPPO CURRICULUM TRAINING")
    print("============================================================")

    for stage_info in curriculum_cfg:
        stage_num = stage_info["stage"]
        num_evs = stage_info["num_evs"]
        episodes = stage_info["episodes"]

        stage_name = f"stage_{stage_num:02d}"
        stage_dir = base_save_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n---> Stage {stage_num}/6: {num_evs} EVs ({episodes} episodes)")
        env = StandardizedEVEnv(num_evs=num_evs, candidate_count=candidate_count, seed=seed + stage_num)

        for ep in range(1, episodes + 1):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_costs = torch.zeros(5)

            for step in range(20):
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
                        "step": step,
                    })

                candidate_stations = env.current_candidates
                graph = builder.build_graph(ev_states, candidate_stations)
                cand_indices = [[j for j in range(len(cand))] for cand in candidate_stations]

                actions, log_probs, values, dist = model(graph, cand_indices)
                actions_np = actions.detach().cpu().numpy()

                next_obs, rewards, dones, truncated, step_info = env.step(actions_np)
                r_val = float(rewards) if isinstance(rewards, (int, float)) else float(np.mean(rewards))
                ep_reward += r_val

                # Compute constraint costs for batch
                step_costs = torch.zeros(5)
                for i, cands in enumerate(candidate_stations):
                    act_i = int(actions_np[i])
                    if act_i < len(cands):
                        c_info = cands[act_i]
                        c_vec = lagrangian.compute_constraint_costs(
                            soc_arrival=float(ev_states[i]["battery_pct"]) / 100.0 - 0.05,
                            free_ports=int(c_info.get("free_ports", 1)),
                            queue_len=int(c_info.get("queue_len", 0)),
                            detour_km=float(c_info.get("distance_km", 1.0)),
                            grid_load_kw=float(c_info.get("grid_load_kw", 10.0)),
                        )
                        step_costs += c_vec
                
                step_costs /= max(1, len(ev_states))
                ep_costs += step_costs

                penalty = lagrangian.get_penalty(step_costs)
                loss = -log_probs.mean() + 0.5 * values.mean() ** 2 + penalty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_ep_costs = ep_costs / 20.0
            mults = lagrangian.update_multipliers(avg_ep_costs)

            history.append({
                "stage": stage_num,
                "episode": ep,
                "reward": ep_reward,
                "loss": loss.item(),
                "lambdas": mults,
            })

            if ep % max(1, episodes // 2) == 0 or ep == episodes:
                print(f"     [Ep {ep:02d}/{episodes:02d}] Reward: {ep_reward:7.1f} | Loss: {loss.item():.4f} | Lambda_1: {mults['lambda_1']:.4f}")

        # Save stage checkpoint
        ckpt_path = stage_dir / "model.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved Checkpoint: {ckpt_path}")

    # Save final history
    history_path = training_log_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n============================================================")
    print("CURRICULUM TRAINING COMPLETED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    train_curriculum()
