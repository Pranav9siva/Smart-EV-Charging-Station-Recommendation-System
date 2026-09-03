from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl_env.gym_ev_charging_env import GymEVChargingEnv

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CHECKPOINT = ROOT / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"
FINETUNED_CHECKPOINT = ROOT / "runs" / "ppo_ckpt" / "ppo_ev_finetuned_50k.zip"
DB_PATH = ROOT / "data" / "stations" / "stations.sqlite"
OUTPUT_DIR = ROOT / "runs" / "evaluation" / "ppo_ab_comparison"
CSV_PATH = OUTPUT_DIR / "ppo_ab_results.csv"
JSON_PATH = OUTPUT_DIR / "ppo_ab_results.json"
MARKDOWN_PATH = OUTPUT_DIR / "ppo_ab_comparison_report.md"
PLOT_DIR = OUTPUT_DIR / "plots"
TRACKED_VEHICLE_COUNT = 10
CANDIDATE_COUNT = 8
EPISODE_COUNT = 3
MAX_STEPS_PER_EPISODE = 10
SEEDS = [42, 43, 44]


def make_env() -> GymEVChargingEnv:
    return GymEVChargingEnv(
        db_path=str(DB_PATH),
        tracked_vehicle_count=TRACKED_VEHICLE_COUNT,
        candidate_count=CANDIDATE_COUNT,
    )


def is_placeholder_candidate(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("station_id") == ""
        and candidate.get("distance_km") == 1e6
        and candidate.get("travel_time_min") == 1e6
        and candidate.get("avg_wait_min") == 1e6
    )


def summarize_episode_metrics(step_rows: list[dict[str, Any]], episode: int, seed: int) -> dict[str, Any]:
    rewards = [float(item.get("reward", 0.0)) for item in step_rows]
    return {
        "episode": episode,
        "seed": seed,
        "steps": len(step_rows),
        "reward_total": round(float(sum(rewards)), 6),
        "reward_mean": round(float(np.mean(rewards)) if rewards else 0.0, 6),
        "reward_std": round(float(np.std(rewards)) if len(rewards) > 1 else 0.0, 6),
        "charging_events": int(sum(1 for item in step_rows if bool(item.get("charging_event")))),
        "successful_charging_decisions": int(sum(1 for item in step_rows if bool(item.get("successful_charging_decision")))),
        "invalid_actions": int(sum(1 for item in step_rows if bool(item.get("invalid_action")))),
        "completed": bool(step_rows[-1].get("completed", False)) if step_rows else False,
        "selected_station_ids": [item.get("selected_station_id") for item in step_rows if item.get("selected_station_id") is not None],
        "distance_km_mean": round(float(np.mean([item.get("distance_km", 0.0) for item in step_rows])) if step_rows else 0.0, 6),
        "waiting_time_mean": round(float(np.mean([item.get("waiting_time_min", 0.0) for item in step_rows])) if step_rows else 0.0, 6),
        "queue_length_mean": round(float(np.mean([item.get("queue_length", 0.0) for item in step_rows])) if step_rows else 0.0, 6),
        "free_ports_mean": round(float(np.mean([item.get("free_ports", 0.0) for item in step_rows])) if step_rows else 0.0, 6),
        "charging_cost_mean": round(float(np.mean([item.get("charging_cost", 0.0) for item in step_rows])) if step_rows else 0.0, 6),
        "grid_load_mean": round(float(np.mean([item.get("grid_load_kw", 0.0) for item in step_rows])) if step_rows else 0.0, 6),
        "battery_soc_mean": round(float(np.mean([item.get("battery_soc", 0.0) for item in step_rows])) if step_rows else 0.0, 6),
    }


def load_model(path: Path) -> PPO:
    env = DummyVecEnv([make_env])
    return PPO.load(str(path), env=env)


def evaluate_model(model_name: str, checkpoint_path: Path, seeds: list[int]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    placeholder_candidates_found = 0
    invalid_actions = 0
    errors: list[str] = []

    try:
        model = load_model(checkpoint_path)
    except Exception as exc:  # pragma: no cover - runtime guard for evaluation
        return {
            "model": model_name,
            "checkpoint": str(checkpoint_path),
            "status": "FAILED",
            "errors": [f"model_load_error:{exc}"],
            "episodes": [],
            "aggregate": {},
            "placeholder_candidates_found": 0,
        }

    for seed in seeds:
        for episode_idx in range(1, EPISODE_COUNT + 1):
            episode_env = make_env()
            try:
                obs, _ = episode_env.reset(seed=seed + episode_idx)
            except Exception as exc:  # pragma: no cover - runtime guard for evaluation
                errors.append(f"reset_error:{model_name}:{seed}:{episode_idx}:{exc}")
                episode_env.close()
                continue

            step_rows: list[dict[str, Any]] = []
            for step_idx in range(MAX_STEPS_PER_EPISODE):
                try:
                    action, _ = model.predict(obs, deterministic=True)
                except Exception as exc:  # pragma: no cover - runtime guard for evaluation
                    errors.append(f"prediction_error:{model_name}:{seed}:{episode_idx}:{step_idx}:{exc}")
                    break

                action_value = int(action)
                if action_value < 0 or action_value >= CANDIDATE_COUNT:
                    invalid_actions += 1
                    errors.append(f"invalid_action:{model_name}:{seed}:{episode_idx}:{step_idx}:{action_value}")
                    break

                try:
                    obs, reward, terminated, truncated, info = episode_env.step(action_value)
                except Exception as exc:  # pragma: no cover - runtime guard for evaluation
                    errors.append(f"step_error:{model_name}:{seed}:{episode_idx}:{step_idx}:{exc}")
                    break

                chosen_station = episode_env.current_candidates[action_value] if action_value < len(episode_env.current_candidates) else None
                placeholder_candidates_found += 1 if chosen_station and is_placeholder_candidate(chosen_station) else 0

                selected_station_id = info.get("chosen_station") or (chosen_station.get("station_id") if chosen_station else None)
                battery_soc = float(obs.get("vehicles", np.zeros((TRACKED_VEHICLE_COUNT, 6), dtype=np.float32))[0, 0]) if isinstance(obs, dict) and "vehicles" in obs else None
                step_row = {
                    "model": model_name,
                    "checkpoint": str(checkpoint_path),
                    "episode": episode_idx,
                    "seed": seed,
                    "step": step_idx + 1,
                    "reward": float(reward),
                    "charging_event": False,
                    "successful_charging_decision": False,
                    "invalid_action": False,
                    "completed": bool(terminated or truncated),
                    "selected_station_id": selected_station_id,
                    "distance_km": float(chosen_station.get("distance_km", 0.0)) if chosen_station else 0.0,
                    "waiting_time_min": float(chosen_station.get("avg_wait_min", 0.0)) if chosen_station else 0.0,
                    "queue_length": int(chosen_station.get("queue_len", 0)) if chosen_station else 0,
                    "free_ports": int(chosen_station.get("free_ports", 0)) if chosen_station else 0,
                    "charging_cost": float(chosen_station.get("price_per_kwh", 0.0)) if chosen_station else 0.0,
                    "grid_load_kw": float(chosen_station.get("grid_load_kw", 0.0)) if chosen_station else 0.0,
                    "battery_soc": battery_soc,
                }
                step_rows.append(step_row)
                results.append(step_row)

                if terminated or truncated:
                    break

            episode_summaries.append(summarize_episode_metrics(step_rows, episode_idx, seed))
            episode_env.close()

    aggregate = {
        "episodes": len(episode_summaries),
        "steps": len(results),
        "reward_mean": round(float(np.mean([item["reward_total"] for item in episode_summaries])) if episode_summaries else 0.0, 6),
        "reward_std": round(float(np.std([item["reward_total"] for item in episode_summaries])) if len(episode_summaries) > 1 else 0.0, 6),
        "charging_events": int(sum(item["charging_events"] for item in episode_summaries)),
        "successful_charging_decisions": int(sum(item["successful_charging_decisions"] for item in episode_summaries)),
        "invalid_actions": int(sum(item["invalid_actions"] for item in episode_summaries)),
        "completed_episodes": int(sum(1 for item in episode_summaries if item["completed"])),
        "placeholder_candidates_found": placeholder_candidates_found,
    }
    return {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "status": "PASSED" if not errors else "FAILED_WITH_WARNINGS",
        "errors": errors,
        "episodes": episode_summaries,
        "aggregate": aggregate,
        "placeholder_candidates_found": placeholder_candidates_found,
        "step_results": results,
    }


def compute_improvement(original: dict[str, Any], fine_tuned: dict[str, Any]) -> dict[str, Any]:
    original_mean = float(original.get("aggregate", {}).get("reward_mean", 0.0))
    fine_mean = float(fine_tuned.get("aggregate", {}).get("reward_mean", 0.0))
    abs_delta = fine_mean - original_mean
    pct_delta = ((fine_mean - original_mean) / original_mean * 100.0) if original_mean else 0.0
    return {
        "reward_abs_delta": round(abs_delta, 6),
        "reward_pct_delta": round(pct_delta, 6),
        "successful_charging_abs_delta": int(fine_tuned.get("aggregate", {}).get("successful_charging_decisions", 0) - original.get("aggregate", {}).get("successful_charging_decisions", 0)),
        "successful_charging_pct_delta": round(
            ((fine_tuned.get("aggregate", {}).get("successful_charging_decisions", 0) - original.get("aggregate", {}).get("successful_charging_decisions", 0)) / max(1, original.get("aggregate", {}).get("successful_charging_decisions", 0))) * 100.0,
            6,
        ),
        "completed_episodes_abs_delta": int(fine_tuned.get("aggregate", {}).get("completed_episodes", 0) - original.get("aggregate", {}).get("completed_episodes", 0)),
        "completed_episodes_pct_delta": round(
            ((fine_tuned.get("aggregate", {}).get("completed_episodes", 0) - original.get("aggregate", {}).get("completed_episodes", 0)) / max(1, original.get("aggregate", {}).get("completed_episodes", 0))) * 100.0,
            6,
        ),
    }


def write_outputs(results: dict[str, Any], comparison: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "model",
            "checkpoint",
            "episode",
            "seed",
            "step",
            "reward",
            "charging_event",
            "successful_charging_decision",
            "invalid_action",
            "completed",
            "selected_station_id",
            "distance_km",
            "waiting_time_min",
            "queue_length",
            "free_ports",
            "charging_cost",
            "grid_load_kw",
            "battery_soc",
        ])
        writer.writeheader()
        for model_name in ("original", "fine_tuned"):
            for row in results[model_name]["step_results"]:
                writer.writerow(row)

    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump({"results": results, "comparison": comparison}, handle, indent=2)

    with MARKDOWN_PATH.open("w", encoding="utf-8") as handle:
        handle.write(comparison["markdown"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rewards_original = [ep["reward_total"] for ep in results["original"]["episodes"]]
        rewards_finetuned = [ep["reward_total"] for ep in results["fine_tuned"]["episodes"]]

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(1, max(len(rewards_original), len(rewards_finetuned)) + 1)
        ax.bar(x - 0.2, rewards_original, width=0.35, label="original")
        ax.bar(x + 0.2, rewards_finetuned, width=0.35, label="fine_tuned")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_ylabel("Reward total")
        ax.set_xlabel("Episode")
        ax.set_title("PPO checkpoint reward comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "reward_comparison.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - runtime guard for plotting
        comparison["plot_errors"] = [f"plot_error:{exc}"]


def run_ab_evaluation() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    original = evaluate_model("original", ORIGINAL_CHECKPOINT, SEEDS)
    fine_tuned = evaluate_model("fine_tuned", FINETUNED_CHECKPOINT, SEEDS)
    comparison = compute_improvement(original, fine_tuned)
    comparison["same_environment_conditions"] = True
    comparison["same_seeds"] = SEEDS
    comparison["same_episode_count"] = EPISODE_COUNT
    comparison["same_steps_per_episode"] = MAX_STEPS_PER_EPISODE
    comparison["same_candidate_count"] = CANDIDATE_COUNT
    comparison["same_tracked_vehicle_count"] = TRACKED_VEHICLE_COUNT
    comparison["db_path"] = str(DB_PATH)
    comparison["placeholder_candidates_used"] = {
        "original": original.get("placeholder_candidates_found", 0),
        "fine_tuned": fine_tuned.get("placeholder_candidates_found", 0),
    }
    comparison["errors"] = []
    comparison["errors"].extend(original.get("errors", []))
    comparison["errors"].extend(fine_tuned.get("errors", []))
    comparison["markdown"] = f"""# PPO A/B Evaluation Report

## Checkpoints
- Original: {ORIGINAL_CHECKPOINT}
- Fine-tuned: {FINETUNED_CHECKPOINT}

## Evaluation Setup
- Database: {DB_PATH}
- Seeds: {SEEDS}
- Episodes per seed: {EPISODE_COUNT}
- Steps per episode: {MAX_STEPS_PER_EPISODE}
- Tracked vehicles: {TRACKED_VEHICLE_COUNT}
- Candidate stations: {CANDIDATE_COUNT}
- Placeholder `_empty_candidate` usage verified: {comparison['placeholder_candidates_used']}

## Aggregate Metrics
- Original reward mean: {original.get('aggregate', {}).get('reward_mean', 0.0)}
- Fine-tuned reward mean: {fine_tuned.get('aggregate', {}).get('reward_mean', 0.0)}
- Reward absolute delta: {comparison['reward_abs_delta']}
- Reward percentage delta: {comparison['reward_pct_delta']}%
- Successful charging decisions delta: {comparison['successful_charging_abs_delta']}
- Completed episodes delta: {comparison['completed_episodes_abs_delta']}

## Episode-Level Summary
"""
    for label, payload in (("original", original), ("fine_tuned", fine_tuned)):
        comparison["markdown"] += f"\n### {label}\n"
        for episode in payload.get("episodes", []):
            comparison["markdown"] += (
                f"- Episode {episode['episode']} seed {episode['seed']}: "
                f"reward_total={episode['reward_total']}, reward_mean={episode['reward_mean']}, "
                f"reward_std={episode['reward_std']}, charging_events={episode['charging_events']}, "
                f"successful_charging_decisions={episode['successful_charging_decisions']}, "
                f"invalid_actions={episode['invalid_actions']}, completed={episode['completed']}\n"
            )
    comparison["markdown"] += f"\n## Validation\n- Same seeds: {comparison['same_seeds']}\n- Same environment conditions: {comparison['same_environment_conditions']}\n- Placeholder candidate usage: {comparison['placeholder_candidates_used']}\n- Errors: {comparison['errors'] if comparison['errors'] else 'none'}\n"

    results = {"original": original, "fine_tuned": fine_tuned}
    write_outputs(results, comparison)
    return {"results": results, "comparison": comparison}


if __name__ == "__main__":
    run_ab_evaluation()
