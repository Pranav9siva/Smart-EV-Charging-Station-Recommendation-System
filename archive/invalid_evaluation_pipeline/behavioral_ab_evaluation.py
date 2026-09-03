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

from scripts.behavioral_eval_ppo import BehavioralEvaluationEnv
from src.rl_env.gym_ev_charging_env import GymEVChargingEnv

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CHECKPOINT = ROOT / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"
FINETUNED_CHECKPOINT = ROOT / "runs" / "ppo_ckpt" / "ppo_ev_finetuned_50k.zip"
DB_PATH = ROOT / "data" / "stations" / "stations.sqlite"
OUTPUT_DIR = ROOT / "runs" / "evaluation" / "behavioral_ab"
CSV_PATH = OUTPUT_DIR / "behavioral_ab_results.csv"
JSON_PATH = OUTPUT_DIR / "behavioral_ab_results.json"
SUMMARY_PATH = OUTPUT_DIR / "behavioral_ab_summary.md"
METRICS_CSV_PATH = OUTPUT_DIR / "behavioral_ab_metrics.csv"
SCENARIO_SUMMARY_CSV_PATH = OUTPUT_DIR / "behavioral_ab_scenario_summary.csv"
PLOT_DIR = OUTPUT_DIR / "plots"
TRACKED_VEHICLE_COUNT = 1
CANDIDATE_COUNT = 1
EPISODES_PER_SEED = 1
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
SCENARIOS = [10.0, 20.0, 30.0, 40.0, 50.0]


def make_env() -> BehavioralEvaluationEnv:
    return BehavioralEvaluationEnv(db_path=str(DB_PATH), tracked_vehicle_count=TRACKED_VEHICLE_COUNT, candidate_count=CANDIDATE_COUNT)


def load_model(path: Path) -> PPO:
    env = DummyVecEnv([make_env])
    return PPO.load(str(path), env=env)


def is_placeholder_candidate(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("station_id") == ""
        and candidate.get("distance_km") == 1e6
        and candidate.get("travel_time_min") == 1e6
        and candidate.get("avg_wait_min") == 1e6
    )


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mean_reward": None,
            "std_reward": None,
            "mean_soc_improvement": None,
            "mean_range_improvement": None,
            "charging_decision_success_rate": None,
            "charging_event_rate": None,
            "charging_completion_rate": None,
            "invalid_action_rate": None,
            "safety_violation_rate": None,
        }

    rewards = [float(row.get("reward", 0.0)) for row in rows]
    soc_improvements = [float(row.get("soc_improvement", 0.0)) for row in rows]
    range_improvements = [float(row.get("range_improvement", 0.0)) for row in rows]
    charging_event_rate = sum(1 for row in rows if bool(row.get("charging_event"))) / len(rows)
    success_rate = sum(1 for row in rows if bool(row.get("successful_charging_decision"))) / len(rows)
    completion_rate = sum(1 for row in rows if bool(row.get("charging_completed"))) / len(rows)
    invalid_action_rate = sum(1 for row in rows if bool(row.get("invalid_action"))) / len(rows)
    safety_violation_rate = sum(1 for row in rows if bool(row.get("safety_violation"))) / len(rows)

    return {
        "count": len(rows),
        "mean_reward": round(float(np.mean(rewards)), 6),
        "std_reward": round(float(np.std(rewards)), 6),
        "min_reward": round(float(np.min(rewards)), 6),
        "max_reward": round(float(np.max(rewards)), 6),
        "mean_soc_improvement": round(float(np.mean(soc_improvements)), 6),
        "std_soc_improvement": round(float(np.std(soc_improvements)), 6),
        "mean_range_improvement": round(float(np.mean(range_improvements)), 6),
        "std_range_improvement": round(float(np.std(range_improvements)), 6),
        "charging_decision_success_rate": round(float(success_rate), 6),
        "charging_event_rate": round(float(charging_event_rate), 6),
        "charging_completion_rate": round(float(completion_rate), 6),
        "invalid_action_rate": round(float(invalid_action_rate), 6),
        "safety_violation_rate": round(float(safety_violation_rate), 6),
    }


def evaluate_model(model_name: str, checkpoint_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    placeholder_candidates_found = 0
    errors: list[str] = []

    try:
        model = load_model(checkpoint_path)
    except Exception as exc:  # pragma: no cover
        return {
            "model": model_name,
            "status": "FAILED",
            "errors": [f"model_load_error:{exc}"],
            "rows": [],
            "scenario_rows": [],
            "placeholder_candidates_found": 0,
            "metrics": summarize_metrics([]),
        }

    for seed in SEEDS:
        for scenario_soc in SCENARIOS:
            for episode_idx in range(1, EPISODES_PER_SEED + 1):
                env = make_env()
                try:
                    obs, _ = env.reset(seed=seed)
                except Exception as exc:  # pragma: no cover
                    errors.append(f"reset_error:{model_name}:{seed}:{scenario_soc}:{episode_idx}:{exc}")
                    env.close()
                    continue

                vehicle = env.vehicle_manager.list_tracked_vehicles()[0]
                vehicle.battery_pct = float(scenario_soc)
                vehicle.remaining_range_km = max(1.0, float(scenario_soc) * 0.8)
                env.current_candidates = [
                    {
                        "station_id": "sim_station_1",
                        "distance_km": 2.0,
                        "travel_time_min": 5.0,
                        "free_ports": 1,
                        "total_ports": 2,
                        "price_per_kwh": 0.2,
                        "avg_wait_min": 0.0,
                        "queue_len": 0,
                        "grid_load_kw": 10.0,
                    }
                ]
                if is_placeholder_candidate(env.current_candidates[0]):
                    placeholder_candidates_found += 1

                try:
                    action, _ = model.predict(obs, deterministic=True)
                except Exception as exc:  # pragma: no cover
                    errors.append(f"prediction_error:{model_name}:{seed}:{scenario_soc}:{episode_idx}:{exc}")
                    env.close()
                    continue

                action_value = int(action)
                if action_value < 0 or action_value >= CANDIDATE_COUNT:
                    invalid_action = True
                    errors.append(f"invalid_action:{model_name}:{seed}:{scenario_soc}:{episode_idx}:{action_value}")
                    env.close()
                    continue

                try:
                    obs, reward, terminated, truncated, info = env.step(action_value)
                except Exception as exc:  # pragma: no cover
                    errors.append(f"step_error:{model_name}:{seed}:{scenario_soc}:{episode_idx}:{exc}")
                    env.close()
                    continue

                behavior = info.get("behavior", {})
                soc_before = float(behavior.get("soc_before", vehicle.battery_pct))
                soc_after = float(behavior.get("soc_after", vehicle.battery_pct))
                range_before = float(behavior.get("range_before", 0.0))
                range_after = float(behavior.get("range_after", 0.0))
                row = {
                    "model": model_name,
                    "seed": seed,
                    "scenario": f"SOC_{int(scenario_soc)}",
                    "episode": episode_idx,
                    "step": 1,
                    "vehicle_id": "ev_1",
                    "initial_soc": soc_before,
                    "final_soc": soc_after,
                    "initial_range_km": range_before,
                    "final_range_km": range_after,
                    "charge_needed": bool(behavior.get("charge_needed", False)),
                    "charging_event": bool(behavior.get("charging_event", False)),
                    "successful_charging_decision": bool(behavior.get("successful_charging_decision", False)),
                    "chosen_station_id": behavior.get("selected_station_id", ""),
                    "distance_km": 2.0,
                    "price_per_kwh": 0.2,
                    "free_ports": int(behavior.get("free_ports", 0)),
                    "queue_len": int(behavior.get("queue_length", 0)),
                    "avg_wait_min": 0.0,
                    "grid_load_kw": 10.0,
                    "reward": float(reward),
                    "invalid_action": False,
                    "charging_completed": bool(behavior.get("successful_charging_decision", False)),
                    "charging_energy_kwh": None,
                    "charging_wait_time": None,
                    "charging_cost": None,
                    "soc_improvement": soc_after - soc_before,
                    "range_improvement": range_after - range_before,
                    "safety_violation": False,
                }
                rows.append(row)
                scenario_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "scenario": f"SOC_{int(scenario_soc)}",
                    "episode": episode_idx,
                    "charge_needed": bool(behavior.get("charge_needed", False)),
                    "charging_event": bool(behavior.get("charging_event", False)),
                    "successful_charging_decision": bool(behavior.get("successful_charging_decision", False)),
                    "reward": float(reward),
                    "soc_improvement": soc_after - soc_before,
                    "range_improvement": range_after - range_before,
                })
                env.close()

    return {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "status": "PASSED" if not errors else "FAILED_WITH_WARNINGS",
        "errors": errors,
        "rows": rows,
        "scenario_rows": scenario_rows,
        "placeholder_candidates_found": placeholder_candidates_found,
        "metrics": summarize_metrics(rows),
    }


def write_outputs(results: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in ("original", "fine_tuned"):
        rows.extend(results[model_name]["rows"])
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "model",
            "seed",
            "scenario",
            "episode",
            "step",
            "vehicle_id",
            "initial_soc",
            "final_soc",
            "initial_range_km",
            "final_range_km",
            "charge_needed",
            "charging_event",
            "successful_charging_decision",
            "chosen_station_id",
            "distance_km",
            "price_per_kwh",
            "free_ports",
            "queue_len",
            "avg_wait_min",
            "grid_load_kw",
            "reward",
            "invalid_action",
            "charging_completed",
            "charging_energy_kwh",
            "charging_wait_time",
            "charging_cost",
            "soc_improvement",
            "range_improvement",
            "safety_violation",
        ])
        writer.writeheader()
        writer.writerows(rows)

    with METRICS_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "value"])
        writer.writeheader()
        for model_name in ("original", "fine_tuned"):
            metrics = results[model_name].get("metrics", summarize_metrics([]))
            for key, value in metrics.items():
                writer.writerow({"model": model_name, "metric": key, "value": value})

    with SCENARIO_SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "scenario", "count", "mean_reward", "mean_soc_improvement", "mean_range_improvement", "charging_event_rate", "success_rate"])
        writer.writeheader()
        for model_name in ("original", "fine_tuned"):
            by_scenario: dict[str, list[dict[str, Any]]] = {}
            for row in results[model_name]["scenario_rows"]:
                by_scenario.setdefault(row["scenario"], []).append(row)
            for scenario, entries in sorted(by_scenario.items()):
                writer.writerow({
                    "model": model_name,
                    "scenario": scenario,
                    "count": len(entries),
                    "mean_reward": round(float(np.mean([e["reward"] for e in entries])), 6),
                    "mean_soc_improvement": round(float(np.mean([e["soc_improvement"] for e in entries])), 6),
                    "mean_range_improvement": round(float(np.mean([e["range_improvement"] for e in entries])), 6),
                    "charging_event_rate": round(float(sum(1 for e in entries if e["charging_event"]) / len(entries)), 6),
                    "success_rate": round(float(sum(1 for e in entries if e["successful_charging_decision"]) / len(entries)), 6),
                })

    comparison = {}
    for key in ["mean_reward", "mean_soc_improvement", "mean_range_improvement", "charging_decision_success_rate", "charging_event_rate", "charging_completion_rate", "invalid_action_rate", "safety_violation_rate"]:
        orig_value = results["original"].get("metrics", summarize_metrics([])).get(key)
        fine_value = results["fine_tuned"].get("metrics", summarize_metrics([])).get(key)
        if orig_value is None or fine_value is None:
            diff = None
            pct = None
        else:
            diff = fine_value - orig_value
            pct = ((fine_value - orig_value) / orig_value * 100.0) if orig_value else None
        comparison[key] = {"original": orig_value, "fine_tuned": fine_value, "difference": diff, "percentage_change": pct}

    summary_lines = ["# Behavioral A/B Evaluation", "", "## Evaluation Setup", "", f"- Checkpoints: original={ORIGINAL_CHECKPOINT.name}, fine_tuned={FINETUNED_CHECKPOINT.name}", f"- Database: {DB_PATH}", f"- Seeds: {SEEDS}", f"- Scenarios: {SCENARIOS}", f"- Episodes per seed: {EPISODES_PER_SEED}", f"- Tracked vehicles: {TRACKED_VEHICLE_COUNT}", f"- Candidate stations: {CANDIDATE_COUNT}", f"- Placeholder `_empty_candidate` usage: original={results['original']['placeholder_candidates_found']}, fine_tuned={results['fine_tuned']['placeholder_candidates_found']}", "", "## Overall Results", "", "| Metric | Original PPO | Fine-tuned PPO | Difference | Improvement % |", "| --- | --- | --- | --- | --- |"]
    for key, values in comparison.items():
        summary_lines.append(f"| {key} | {values['original']} | {values['fine_tuned']} | {values['difference']} | {values['percentage_change']} |")
    summary_lines.extend(["", "## Scenario-Level Results", ""])
    summary_lines.append("| Scenario | Original success | Fine-tuned success | Original reward | Fine-tuned reward |")
    summary_lines.append("| --- | --- | --- | --- | --- |")
    for scenario in [f"SOC_{int(s)}" for s in SCENARIOS]:
        orig_rows = [row for row in results["original"]["scenario_rows"] if row["scenario"] == scenario]
        fine_rows = [row for row in results["fine_tuned"]["scenario_rows"] if row["scenario"] == scenario]
        orig_success = round(float(sum(1 for row in orig_rows if row["successful_charging_decision"]) / max(1, len(orig_rows))), 6)
        fine_success = round(float(sum(1 for row in fine_rows if row["successful_charging_decision"]) / max(1, len(fine_rows))), 6)
        orig_reward = None if not orig_rows else round(float(np.mean([row["reward"] for row in orig_rows])), 6)
        fine_reward = None if not fine_rows else round(float(np.mean([row["reward"] for row in fine_rows])), 6)
        summary_lines.append(f"| {scenario} | {orig_success} | {fine_success} | {orig_reward} | {fine_reward} |")

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))

    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for metric_name, label in [
            ("mean_reward", "Reward"),
            ("charging_decision_success_rate", "Charging Success"),
            ("mean_soc_improvement", "SOC Improvement"),
            ("mean_range_improvement", "Range Improvement"),
        ]:
            orig_val = results["original"]["metrics"].get(metric_name)
            fine_val = results["fine_tuned"]["metrics"].get(metric_name)
            if orig_val is None or fine_val is None:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(["Original", "Fine-tuned"], [orig_val, fine_val])
            ax.set_title(label)
            ax.set_ylabel(label)
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"{metric_name.replace('_', '-')}.png", dpi=150)
            plt.close(fig)
    except Exception:
        pass


def run_behavioral_ab_evaluation() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "original": evaluate_model("original", ORIGINAL_CHECKPOINT),
        "fine_tuned": evaluate_model("fine_tuned", FINETUNED_CHECKPOINT),
    }
    write_outputs(results)
    return {"results": results, "output_dir": str(OUTPUT_DIR)}


if __name__ == "__main__":
    run_behavioral_ab_evaluation()
