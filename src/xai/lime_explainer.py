from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LIMEExplainer:
    """Lightweight LIME-style explanation support for recommendations."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir or "outputs/lime_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def explain(self, recommendation: dict[str, Any], features: dict[str, float] | None = None) -> dict[str, Any]:
        feature_values = features or {
            "battery": float(recommendation.get("battery_pct", 0.0) or 0.0),
            "distance": float(recommendation.get("travel_distance", 0.0) or 0.0),
            "waiting_time": float(recommendation.get("waiting_time", 0.0) or 0.0),
        }
        payload = {
            "selected_station": recommendation.get("station"),
            "local_explanation": {k: round(v, 3) for k, v in feature_values.items()},
            "method": "lime-like-approximation",
        }
        self._write_json(payload, "lime_report.json")
        return payload

    def _write_json(self, payload: dict[str, Any], filename: str) -> None:
        (self.output_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
