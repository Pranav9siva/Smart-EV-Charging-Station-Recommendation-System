from __future__ import annotations

from typing import Any


def run_behavioral_ab_evaluation() -> dict[str, Any]:
	from scripts.legacy.behavioral_ab_evaluation import run_behavioral_ab_evaluation as implementation

	return implementation()