from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class PPOEvaluator:
    """Lightweight evaluator for PPO recommendation output quality."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir or "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        if not recommendations:
            return {
                "average_reward": 0.0,
                "success_rate": 0.0,
                "episode_length": 0,
                "queue_reduction": 0.0,
                "waiting_time": 0.0,
                "charging_cost": 0.0,
                "energy_consumption": 0.0,
            }

        rewards = [float(item.get("ppo_reward", 0.0) or 0.0) for item in recommendations]
        success_rate = sum(1 for item in recommendations if item.get("selected_station")) / max(1, len(recommendations))
        episode_length = len(recommendations)
        queue_reduction = sum(float(item.get("queue_length", 0) or 0) for item in recommendations) / max(1, len(recommendations))
        waiting_time = sum(float(item.get("waiting_time", 0.0) or 0.0) for item in recommendations) / max(1, len(recommendations))
        charging_cost = sum(float(item.get("charging_cost", 0.0) or 0.0) for item in recommendations) / max(1, len(recommendations))
        energy_consumption = sum(float(item.get("energy_consumption", 0.0) or 0.0) for item in recommendations) / max(1, len(recommendations))

        report = {
            "average_reward": round(sum(rewards) / max(1, len(rewards)), 4),
            "success_rate": round(success_rate, 4),
            "episode_length": int(episode_length),
            "queue_reduction": round(queue_reduction, 4),
            "waiting_time": round(waiting_time, 4),
            "charging_cost": round(charging_cost, 4),
            "energy_consumption": round(energy_consumption, 4),
        }
        self._write_outputs(report, recommendations)
        return report

    def _write_outputs(self, report: dict[str, Any], recommendations: list[dict[str, Any]]) -> None:
        report_path = self.output_dir / "evaluation_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        csv_path = self.output_dir / "evaluation_report.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report.keys()))
            writer.writeheader()
            writer.writerow(report)

        html_path = self.output_dir / "evaluation_report.html"
        html_path.write_text(
            "<html><body><h1>Evaluation Report</h1><pre>" + json.dumps(report, indent=2) + "</pre></body></html>",
            encoding="utf-8",
        )

        episode_path = self.output_dir / "episode_statistics.csv"
        with episode_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "selected_station", "ppo_reward", "queue_length", "waiting_time", "charging_cost"])
            writer.writeheader()
            for index, item in enumerate(recommendations):
                writer.writerow({
                    "step": index,
                    "selected_station": item.get("selected_station", ""),
                    "ppo_reward": item.get("ppo_reward", 0.0),
                    "queue_length": item.get("queue_length", 0),
                    "waiting_time": item.get("waiting_time", 0.0),
                    "charging_cost": item.get("charging_cost", 0.0),
                })
