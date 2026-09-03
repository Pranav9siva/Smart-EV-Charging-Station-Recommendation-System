from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Gauge, Histogram, REGISTRY as PROMETHEUS_REGISTRY, generate_latest

REGISTRY = PROMETHEUS_REGISTRY

_METRICS: dict[str, Any] = {}
_METRIC_TYPES: dict[str, str] = {}


def _get_or_create_metric(name: str, metric_type: str, description: str, labels: tuple[str, ...] = ()) -> Any:
    key = (name, labels)
    if key not in _METRICS:
        if metric_type == "counter":
            metric = Counter(name, description, labelnames=labels, registry=REGISTRY)
        elif metric_type == "gauge":
            metric = Gauge(name, description, labelnames=labels, registry=REGISTRY)
        elif metric_type == "histogram":
            metric = Histogram(name, description, labelnames=labels, registry=REGISTRY)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        _METRICS[key] = metric
        _METRIC_TYPES[name] = metric_type
    return _METRICS[key]


def init_metrics() -> None:
    for name, metric_type, description in [
        ("simulation_steps_total", "counter", "Total simulation steps executed"),
        ("simulation_running", "gauge", "Whether the simulation is currently running"),
        ("simulation_duration_seconds", "gauge", "Elapsed simulation duration in seconds"),
        ("simulation_speed", "gauge", "Simulation speed metric"),
        ("simulation_fps", "gauge", "Simulation frames per second"),
        ("simulation_step", "gauge", "Current simulation step"),
        ("episode_number", "gauge", "Current episode number"),
        ("active_vehicles", "gauge", "Number of active vehicles"),
        ("charging_vehicles", "gauge", "Number of vehicles currently charging"),
        ("waiting_vehicles", "gauge", "Number of vehicles waiting"),
        ("low_battery_vehicles", "gauge", "Number of vehicles with low battery"),
        ("successful_recommendations_total", "counter", "Successful recommendations"),
        ("failed_recommendations_total", "counter", "Failed recommendations"),
        ("charging_requests", "counter", "Charging requests received"),
        ("successful_charging", "counter", "Successful charging completions"),
        ("rejected_charging", "counter", "Rejected charging requests"),
        ("station_utilization_percent", "gauge", "Station utilization percent"),
        ("available_ports", "gauge", "Number of available ports"),
        ("occupied_ports", "gauge", "Number of occupied ports"),
        ("average_wait_time", "gauge", "Average wait time"),
        ("average_queue_length", "gauge", "Average queue length"),
        ("average_charging_time", "gauge", "Average charging time"),
        ("charging_queue_length", "gauge", "Current charging queue length"),
        ("station_utilization", "gauge", "Station utilization ratio"),
        ("battery_distribution", "gauge", "Battery distribution metric"),
        ("average_soc", "gauge", "Average state of charge"),
        ("vehicle_count", "gauge", "Current vehicle count"),
        ("finished_trips", "counter", "Finished trips"),
        ("distance_travelled", "gauge", "Distance travelled"),
        ("average_speed", "gauge", "Average vehicle speed"),
        ("traffic_density", "gauge", "Traffic density metric"),
        ("ppo_reward", "gauge", "Current PPO reward"),
        ("ppo_reward_avg", "gauge", "Average PPO reward"),
        ("episode_reward", "gauge", "Episode reward"),
        ("episode_length", "gauge", "Episode length"),
        ("policy_inference_latency", "gauge", "Policy inference latency"),
        ("decision_time", "gauge", "Decision time"),
        ("cpu_usage", "gauge", "CPU usage percent"),
        ("memory_usage", "gauge", "Memory usage percent"),
        ("gpu_usage", "gauge", "GPU usage percent"),
        ("disk_usage", "gauge", "Disk usage percent"),
        ("websocket_connections", "gauge", "Number of active websocket connections"),
        ("api_requests_total", "counter", "Total API requests"),
        ("api_response_time", "histogram", "API response time"),
        ("api_latency_ms", "gauge", "API latency in milliseconds"),
        ("errors_total", "counter", "Total errors"),
        ("warnings_total", "counter", "Total warnings"),
    ]:
        _get_or_create_metric(name, metric_type, description)


init_metrics()


def update_metric(name: str, value: float | int, labels: tuple[str, ...] = ()) -> None:
    metric_type = _METRIC_TYPES.get(name, "gauge")
    if metric_type == "counter":
        metric = _get_or_create_metric(name, metric_type, f"{name} counter")
        metric.labels(*labels).inc(float(value)) if labels else metric.inc(float(value))
        return
    metric = _get_or_create_metric(name, metric_type, f"{name} metric")
    metric.labels(*labels).set(float(value)) if labels else metric.set(float(value))


def increment_counter(name: str, value: float | int = 1, labels: tuple[str, ...] = ()) -> None:
    metric = _get_or_create_metric(name, "counter", f"{name} counter")
    metric.labels(*labels).inc(float(value)) if labels else metric.inc(float(value))


def observe_histogram(name: str, value: float | int, labels: tuple[str, ...] = ()) -> None:
    metric = _get_or_create_metric(name, "histogram", f"{name} histogram")
    metric.labels(*labels).observe(float(value)) if labels else metric.observe(float(value))


def get_metrics_payload() -> str:
    return generate_latest(REGISTRY).decode("utf-8")
