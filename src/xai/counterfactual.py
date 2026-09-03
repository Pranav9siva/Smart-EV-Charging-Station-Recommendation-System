from __future__ import annotations

from typing import Any


class CounterfactualExplainer:
    """Generate simple counterfactual alternatives for a recommendation."""

    def build(self, recommendation: dict[str, Any]) -> dict[str, Any]:
        station = recommendation.get("station") or "S1"
        return {
            "selected_station": station,
            "counterfactuals": [
                {"station_id": "S2", "reason": "lower queue"},
                {"station_id": "S3", "reason": "lower price"},
            ],
        }
