# Monitoring Guide

## Architecture

The monitoring layer is a thin extension over the existing FastAPI app. It exposes Prometheus metrics, health endpoints, and a Grafana dashboard without changing the business logic or RL pipeline.

## Metrics

The monitoring layer is now integrated with the live runtime events from the simulation controller, station manager, vehicle manager, RL environment, recommendation engine, and visualization service.

The following metrics are exposed at `/metrics`:

- Simulation: `simulation_steps_total`, `simulation_running`, `simulation_duration_seconds`, `simulation_speed`, `simulation_fps`, `simulation_step`
- Episodes: `episode_number`, `episode_reward`, `episode_length`, `ppo_reward`, `ppo_reward_avg`
- Vehicles: `active_vehicles`, `charging_vehicles`, `waiting_vehicles`, `low_battery_vehicles`, `vehicle_count`, `average_soc`, `average_speed`, `distance_travelled`, `traffic_density`
- Charging: `charging_requests`, `successful_charging`, `rejected_charging`, `charging_queue_length`, `average_queue_length`, `average_charging_time`, `available_ports`, `occupied_ports`, `average_wait_time`, `station_utilization_percent`, `station_utilization`
- RL: `policy_inference_latency`, `decision_time`
- System: `cpu_usage`, `memory_usage`, `gpu_usage`, `disk_usage`, `websocket_connections`, `api_requests_total`, `api_response_time`, `api_latency_ms`, `errors_total`, `warnings_total`

## Prometheus Setup

Run Prometheus with:

```bash
docker compose -f docker-compose.monitoring.yml up -d prometheus
```

## Grafana Setup

Open Grafana at `http://localhost:3000` and import `grafana/dashboard.json`.
The dashboard includes panels for simulation performance, vehicle state, charging station health, RL behavior, API metrics, and system resources.

## Docker Commands

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

## Troubleshooting

- If `/metrics` does not return data, verify that the app is running and that `prometheus_client` is installed.
- If Grafana does not show panels, import the dashboard JSON manually.
