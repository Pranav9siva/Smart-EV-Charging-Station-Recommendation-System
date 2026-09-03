from __future__ import annotations

from typing import Any


class DecisionTrace:
    """Create a structured decision trace for explainability."""

    def build(self, recommendation: dict[str, Any]) -> dict[str, Any]:
        return {
            "selected_station": recommendation.get("station"),
            "confidence": recommendation.get("confidence", 0.0),
            "policy_probability": recommendation.get("policy_probability", 0.0),
            "steps": [
                {"step": 1, "event": "observation captured"},
                {"step": 2, "event": "policy evaluated candidate stations"},
                {"step": 3, "event": "station selected"},
            ],
        }
