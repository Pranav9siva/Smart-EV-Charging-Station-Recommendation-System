from __future__ import annotations

from typing import Any


def battery_color(battery_pct: float | None) -> str:
    if battery_pct is None:
        return "#6c757d"
    if battery_pct >= 70:
        return "#2e8b57"
    if battery_pct >= 35:
        return "#f4b400"
    return "#d9534f"


def station_status_color(status: str | None) -> str:
    status_key = (status or "available").lower()
    palette = {
        "available": "#2e8b57",
        "busy": "#f4b400",
        "charging": "#1f77b4",
        "offline": "#6c757d",
    }
    return palette.get(status_key, "#2e8b57")


def station_status_from_metrics(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "offline"
    available = int(metrics.get("available_ports", 0) or 0)
    if available <= 0:
        return "busy"
    if metrics.get("charging"):
        return "charging"
    return "available"
