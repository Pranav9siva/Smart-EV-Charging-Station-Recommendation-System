from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.evaluator import PPOEvaluator


class BenchmarkRunner:
    """Simple benchmark runner for research evaluation."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir or "outputs")
        self.evaluator = PPOEvaluator(self.output_dir)

    def run(self, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        return self.evaluator.evaluate(recommendations)
