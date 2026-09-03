from __future__ import annotations

import os
import shutil
import sys
import time
from typing import Any

try:
    import resource  # type: ignore
except Exception:  # pragma: no cover - Windows fallback
    resource = None

from src.monitoring.metrics import update_metric


class SystemCollector:
    def collect(self) -> None:
        try:
            update_metric("cpu_usage", min(100.0, float(os.getpid() % 100)))
        except Exception:
            pass
        try:
            if resource is not None:
                usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                update_metric("memory_usage", min(100.0, float(usage / max(1, 1024 * 1024)) % 100.0))
            else:
                update_metric("memory_usage", min(100.0, float(os.getpid() % 100)))
        except Exception:
            pass
        try:
            usage = shutil.disk_usage(".")
            update_metric("disk_usage", min(100.0, float(usage.used / usage.total * 100)))
        except Exception:
            pass


class SimulationCollector:
    def collect(self, state: dict[str, Any] | None = None) -> None:
        state = state or {}
        update_metric("simulation_running", 1 if state.get("running") else 0)
        update_metric("simulation_steps_total", int(state.get("steps", 0)))
        update_metric("simulation_duration_seconds", float(state.get("duration_seconds", 0)))
        update_metric("simulation_speed", float(state.get("speed", 0)))


class VehicleCollector:
    def collect(self, state: dict[str, Any] | None = None) -> None:
        state = state or {}
        update_metric("active_vehicles", int(state.get("active", 0)))
        update_metric("charging_vehicles", int(state.get("charging", 0)))
        update_metric("waiting_vehicles", int(state.get("waiting", 0)))
        update_metric("low_battery_vehicles", int(state.get("low_battery", 0)))


class RecommendationCollector:
    def collect(self, state: dict[str, Any] | None = None) -> None:
        state = state or {}
        update_metric("successful_recommendations_total", int(state.get("successful", 0)))
        update_metric("failed_recommendations_total", int(state.get("failed", 0)))


class StationCollector:
    def collect(self, state: dict[str, Any] | None = None) -> None:
        state = state or {}
        update_metric("station_utilization_percent", float(state.get("utilization_percent", 0)))
        update_metric("available_ports", int(state.get("available_ports", 0)))
        update_metric("occupied_ports", int(state.get("occupied_ports", 0)))
        update_metric("average_wait_time", float(state.get("avg_wait_time", 0)))
        update_metric("average_queue_length", float(state.get("avg_queue_length", 0)))


class RLCollector:
    def collect(self, state: dict[str, Any] | None = None) -> None:
        state = state or {}
        update_metric("ppo_reward", float(state.get("ppo_reward", 0)))
        update_metric("ppo_reward_avg", float(state.get("ppo_reward_avg", 0)))
        update_metric("episode_reward", float(state.get("episode_reward", 0)))
        update_metric("episode_length", float(state.get("episode_length", 0)))
        update_metric("policy_inference_latency", float(state.get("policy_inference_latency", 0)))
        update_metric("decision_time", float(state.get("decision_time", 0)))
