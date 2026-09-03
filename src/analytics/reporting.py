from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class AnalyticsEngine:
    """Generate daily/weekly/monthly summaries for research-grade reports."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir or "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_reports(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        summary = {
            "total_records": len(records),
            "daily_report": {"records": len(records)},
            "weekly_report": {"records": len(records)},
            "monthly_report": {"records": len(records)},
            "simulation_summary": {"records": len(records)},
            "station_efficiency": {"records": len(records)},
            "vehicle_statistics": {"records": len(records)},
            "recommendation_accuracy": {"records": len(records)},
            "charging_efficiency": {"records": len(records)},
            "energy_consumption": sum(float(item.get("energy_consumption", 0.0) or 0.0) for item in records),
            "revenue_estimation": sum(float(item.get("charging_cost", 0.0) or 0.0) for item in records),
            "co2_savings": round(sum(float(item.get("energy_consumption", 0.0) or 0.0) for item in records) * 0.4, 3),
        }

        self._write_json(summary, "analytics_summary.json")
        self._write_csv(summary, "analytics_summary.csv")
        return summary

    def _write_json(self, payload: dict[str, Any], filename: str) -> None:
        (self.output_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_csv(self, payload: dict[str, Any], filename: str) -> None:
        with (self.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
            writer.writeheader()
            writer.writerow(payload)
