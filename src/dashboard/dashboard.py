from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

REQUIRED_STATION_FIELDS = {
    "station_id",
    "name",
    "latitude",
    "longitude",
    "x",
    "y",
    "total_ports",
    "available_ports",
    "occupied_ports",
    "queue",
    "price_per_kwh",
    "waiting_time_min",
    "grid_load_kw",
    "utilization",
}

REQUIRED_VEHICLE_FIELDS = {
    "vehicle_id",
    "x",
    "y",
    "speed",
    "battery",
    "state",
    "current_station",
    "recommended_station",
    "recommendation_score",
    "reward",
    "queue",
    "price",
    "waiting_time_min",
    "available_ports",
    "grid_load_kw",
}

OPTIONAL_COMPATIBILITY_FIELDS = {
    "station_id",
    "name",
    "latitude",
    "longitude",
    "x",
    "y",
    "total_ports",
    "available_ports",
    "occupied_ports",
    "queue",
    "price_per_kwh",
    "waiting_time_min",
    "grid_load_kw",
    "utilization",
    "vehicle_id",
    "speed",
    "battery",
    "state",
    "current_station",
    "recommended_station",
    "recommendation_score",
    "reward",
    "queue",
    "price",
    "waiting_time_min",
    "available_ports",
    "grid_load_kw",
}

REQUIRED_TRAFFIC_FIELDS = {"average_speed", "traffic_density", "congestion_percent", "vehicle_count"}

REQUIRED_PPO_FIELDS = {"vehicle_id", "action", "station_id", "reward", "battery", "decision_time", "decision_features"}


class Dashboard:
    """Lightweight file-backed dashboard.

    Usage:
      - Instantiate: `db = Dashboard()`
      - Call `db.update(state)` every simulation step where `state` is a dict containing
        `vehicles` (list) and `stations` (list) and `tracked_ids` (list of vehicle ids).
      - Serve the `outputs/` directory (e.g. `python -m http.server` inside project root or use `scripts/run_dashboard.py`).

    The dashboard writes `dashboard_state.json`, `dashboard.html`, and a richer
    `research_dashboard.html` into the outputs folder so that live simulation metrics
    can be inspected without changing the backend logic.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        fleet_size: int = 1000,
        station_count: int = 500,
        tracked: int = 10,
        dashboard_state_interval: int = 1,
        dashboard_report_interval: int = 10,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fleet_size = int(fleet_size)
        self.station_count = int(station_count)
        self.tracked = int(tracked)
        self.dashboard_state_interval = max(1, int(dashboard_state_interval))
        self.dashboard_report_interval = max(1, int(dashboard_report_interval))
        self._update_count = 0
        self._last_state: Dict[str, Any] = self._sample_state()

        # initial empty state
        self.state_path = self.output_dir / "dashboard_state.json"
        self.history_path = self.output_dir / "dashboard_history.json"
        self.html_path = self.output_dir / "dashboard.html"
        self.research_dashboard_path = self.output_dir / "research_dashboard.html"
        self.summary_path = self.output_dir / "dashboard_summary.json"
        # Always refresh dashboard HTML so frontend contract updates are applied
        # even when a stale artifact already exists in outputs/.
        self._write_html()
        if not self.research_dashboard_path.exists():
          self._write_research_dashboard(self._last_state)
        if not self.state_path.exists():
          self._write_state(self._last_state)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        tmp_path.replace(path)

    def _atomic_write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(path)

    def _write_state(self, state: Dict[str, Any]) -> None:
        self._atomic_write_json(self.state_path, state)

    def _write_history(self, state: Dict[str, Any]) -> None:
        history: Dict[str, Any] = {"schema_version": 1, "entries": []}
        if self.history_path.exists():
            try:
                previous = json.loads(self.history_path.read_text(encoding="utf-8"))
                if isinstance(previous, dict) and isinstance(previous.get("entries"), list):
                    history = previous
            except Exception:
                history = {"schema_version": 1, "entries": []}
        history.setdefault("entries", [])
        history["entries"].append(state)
        history["entries"] = history["entries"][-200:]
        self._atomic_write_json(self.history_path, history)

    def update(self, state: Dict[str, Any]) -> None:
        """Update dashboard with the simulation state using one canonical schema."""
        out = self._build_canonical_state(state)
        self._last_state = out
        self._update_count += 1
        self._write_state(out)
        self._write_history(out)
        self._write_html()
        if self._update_count % self.dashboard_report_interval == 0:
            self._write_research_dashboard(out)
            self._write_summary(out.get("summary") or {})

    def _build_canonical_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raw_vehicles = state.get("vehicles")
        simulation_state = self._build_simulation_state(state)
        vehicles_raw_list = cast(list[Any], raw_vehicles) if isinstance(raw_vehicles, list) else []
        vehicles: list[dict[str, Any]] = [item for item in vehicles_raw_list if isinstance(item, dict)]
        raw_stations = state.get("station_details")
        if not isinstance(raw_stations, list):
            raw_stations = state.get("stations")
        stations_raw_list = cast(list[Any], raw_stations) if isinstance(raw_stations, list) else []
        stations: list[dict[str, Any]] = [item for item in stations_raw_list if isinstance(item, dict)]
        raw_traffic = state.get("traffic")
        if isinstance(raw_traffic, dict):
            traffic_raw_list = [raw_traffic]
        elif isinstance(raw_traffic, list):
            traffic_raw_list = cast(list[Any], raw_traffic)
        else:
            traffic_raw_list = []
        traffic_items = [item for item in traffic_raw_list if isinstance(item, dict)]
        raw_ppo = state.get("ppo")
        if isinstance(raw_ppo, dict):
            ppo_items = [raw_ppo]
        elif isinstance(raw_ppo, list):
            ppo_items = [item for item in raw_ppo if isinstance(item, dict)]
        else:
            ppo_items = []
        raw_metrics = state.get("metrics")
        metrics_raw_list = cast(list[Any], raw_metrics) if isinstance(raw_metrics, list) else []
        metrics: list[dict[str, Any]] = [item for item in metrics_raw_list if isinstance(item, dict)]
        raw_summary = state.get("summary")
        summary = cast(dict[str, Any], raw_summary) if isinstance(raw_summary, dict) else self._build_summary(vehicles, stations, metrics)
        raw_recommendations = state.get("recommendations")
        recommendations_raw_list = cast(list[Any], raw_recommendations) if isinstance(raw_recommendations, list) else []
        recommendations: list[dict[str, Any]] = [item for item in recommendations_raw_list if isinstance(item, dict)]

        canonical_stations = self._canonicalize_stations(stations)
        canonical_vehicles = self._canonicalize_vehicles(vehicles)
        canonical_traffic = self._canonicalize_traffic(traffic_items, state.get("simulation"), state.get("kpis"))
        canonical_ppo = self._canonicalize_ppo(ppo_items)
        validation = self._validate_dashboard_payload(canonical_stations, canonical_vehicles, canonical_traffic, canonical_ppo)

        fleet_vehicles: list[dict[str, Any]] = []
        for index in range(max(0, int(self.fleet_size))):
            if index < len(canonical_vehicles):
                fleet_vehicles.append(canonical_vehicles[index])
            else:
                fleet_vehicles.append({
                    "vehicle_id": f"vehicle_{index + 1}",
                    "x": 0.0,
                    "y": 0.0,
                    "speed": 0.0,
                    "battery": 0.0,
                    "state": "unknown",
                    "current_station": "N/A",
                    "recommended_station": "N/A",
                    "recommendation_score": 0.0,
                    "reward": 0.0,
                    "queue": 0,
                    "price": 0.0,
                    "waiting_time_min": 0.0,
                    "available_ports": 0,
                    "grid_load_kw": 0.0,
                })

        tracked_id_values = state.get("tracked_ids") or []
        tracked_ids = [str(item) for item in tracked_id_values if str(item) not in {"", "None", "N/A"}]
        if not tracked_ids:
            tracked_ids = [str(item.get("vehicle_id")) for item in canonical_vehicles if item.get("vehicle_id") is not None]
        tracked_ids_set = set(tracked_ids)
        for item in canonical_vehicles:
            item["tracked"] = str(item.get("vehicle_id")) in tracked_ids_set
        if canonical_vehicles and not tracked_ids_set:
            canonical_vehicles[0]["tracked"] = True

        simulation_payload = {
            "running": bool(simulation_state.get("running", True)),
            "paused": bool(simulation_state.get("paused") if "paused" in simulation_state else False) or str(simulation_state.get("status") or "").lower() in {"paused", "pause", "stopped"},
            "step": simulation_state.get("step"),
            "sim_time": simulation_state.get("time"),
            "time": simulation_state.get("time"),
            "speed": simulation_state.get("speed"),
            "fleet_size": simulation_state.get("fleet_size") or simulation_state.get("ev_count") or len(canonical_vehicles),
            "ev_count": simulation_state.get("ev_count") or len(canonical_vehicles),
            "active_vehicles": len(canonical_vehicles),
            "tracked_count": len([item for item in canonical_vehicles if bool(item.get("tracked"))]),
            "congestion": canonical_traffic.get("congestion_percent"),
            "congestion_percent": canonical_traffic.get("congestion_percent"),
            "connection_status": simulation_state.get("connection_status") or "LIVE",
        }
        simulation_payload["paused"] = bool(simulation_payload["paused"] or not simulation_payload["running"])
        if simulation_state.get("simulation_status") in {"PAUSED", "Paused", "paused"}:
            simulation_payload["paused"] = True
            simulation_payload["running"] = False
        elif simulation_state.get("simulation_status") in {"COMPLETED", "FINISHED", "Completed", "Finished", "completed", "finished"}:
            simulation_payload["paused"] = False
            simulation_payload["running"] = False
        elif simulation_state.get("simulation_status") in {"ERROR", "Error", "error"}:
            simulation_payload["paused"] = True
            simulation_payload["running"] = False
        elif simulation_state.get("running") is None and simulation_state.get("status") is None:
            simulation_payload["running"] = True
        simulation_status = simulation_state.get("simulation_status") or simulation_state.get("status")
        if not simulation_status:
            simulation_status = "PAUSED" if simulation_payload["paused"] else "RUNNING"
        elif isinstance(simulation_status, str):
            simulation_status = simulation_status.upper()
            if simulation_status == "FINISHED":
                simulation_status = "COMPLETED"
            if simulation_status not in {"RUNNING", "PAUSED", "COMPLETED", "ERROR"}:
                simulation_status = "RUNNING" if simulation_payload["running"] else "PAUSED"
        simulation_payload["simulation_status"] = simulation_status

        canonical_simulation = self._build_canonical_simulation(state, simulation_payload, canonical_vehicles)
        simulation_payload.update({
            "step": canonical_simulation.get("step", simulation_payload.get("step")),
            "time": canonical_simulation.get("time", simulation_payload.get("time")),
            "sim_time": canonical_simulation.get("time", simulation_payload.get("time")),
            "status": canonical_simulation.get("status", simulation_payload.get("status") or simulation_status),
            "speed": canonical_simulation.get("speed", simulation_payload.get("speed")),
            "vehicle_count": canonical_simulation.get("vehicle_count", simulation_payload.get("vehicle_count") or simulation_payload.get("ev_count") or len(canonical_vehicles)),
            "ev_count": canonical_simulation.get("ev_count", simulation_payload.get("ev_count") or len(canonical_vehicles)),
            "connection": canonical_simulation.get("connection", simulation_payload.get("connection_status") or "LIVE"),
            "connection_status": canonical_simulation.get("connection", simulation_payload.get("connection_status") or "LIVE"),
            "running": bool(canonical_simulation.get("running", simulation_payload.get("running", True))),
            "paused": bool(canonical_simulation.get("paused", simulation_payload.get("paused", False))),
            "simulation_status": canonical_simulation.get("simulation_status", simulation_payload.get("simulation_status")),
        })

        network_payload = {
            "average_speed": canonical_traffic.get("average_speed"),
            "traffic_level": self._derive_traffic_level(canonical_traffic.get("congestion_percent")),
            "congestion_percent": canonical_traffic.get("congestion_percent"),
            "total_energy": self._coerce_float(state.get("kpis", {}).get("energy_consumed_kwh") if isinstance(state.get("kpis"), dict) else None),
            "stations": canonical_stations,
            "vehicles": canonical_vehicles,
        }

        latest_recommendation = state.get("latest_recommendation")
        ppo_live = self._build_latest_ppo(canonical_ppo, latest_recommendation)
        history_payload = self._build_history_payload(canonical_ppo, recommendations, canonical_stations)
        station_payload = self._build_station_payload(canonical_stations, ppo_live)
        tracked_ev_payload = self._build_tracked_ev_payload(canonical_vehicles, ppo_live, latest_recommendation)
        stations_payload = self._build_station_summary_payload(canonical_stations)
        vehicles_payload = self._build_vehicle_summary_payload(canonical_vehicles)
        canonical_tracked_ev = self._build_canonical_tracked_ev(state, tracked_ev_payload, canonical_vehicles, ppo_live, latest_recommendation)
        canonical_ppo_decision = self._build_canonical_ppo_decision(state, ppo_live, latest_recommendation, canonical_ppo)
        canonical_station = self._build_canonical_station(state, station_payload, canonical_stations, ppo_live)
        canonical_vehicle_list = self._build_canonical_vehicle_list(state, canonical_vehicles)
        canonical_station_list = self._build_canonical_station_list(state, canonical_stations)
        canonical_history = self._build_canonical_history(state, history_payload, recommendations, canonical_ppo)
        ppo_payload = [dict(item) for item in canonical_ppo]
        if not ppo_payload and ppo_live and ppo_live.get("selected_station") not in {None, "N/A"}:
            ppo_payload = [ppo_live]
        if ppo_payload and isinstance(ppo_payload[0], dict):
            ppo_payload[0].setdefault("ev_id", ppo_payload[0].get("vehicle_id") or ppo_payload[0].get("ev_id"))
            ppo_payload[0].setdefault("selected_station", ppo_payload[0].get("station_id") or ppo_payload[0].get("selected_station"))
            ppo_payload[0].setdefault("reason", ppo_payload[0].get("decision_features") or ppo_payload[0].get("reason"))

        charts_payload = self._build_charts_payload(canonical_ppo, recommendations, canonical_stations)
        station_statistics = self._build_station_statistics(canonical_stations)

        telemetry_timestamp = int(state.get("telemetry", {}).get("timestamp") if isinstance(state.get("telemetry"), dict) else 0)
        if telemetry_timestamp <= 0:
            telemetry_timestamp = int(state.get("simulation", {}).get("timestamp") if isinstance(state.get("simulation"), dict) and isinstance(state.get("simulation", {}).get("timestamp"), (int, float)) else 0)
        if telemetry_timestamp <= 0:
            telemetry_timestamp = int(time.time() * 1000)
        telemetry_age_ms = max(0, int(time.time() * 1000) - telemetry_timestamp)
        telemetry_status = "LIVE"
        if telemetry_age_ms > 5000:
            telemetry_status = "STALE"
        if telemetry_age_ms > 20000:
            telemetry_status = "DISCONNECTED"
        kpis_payload = dict(state.get("kpis") or {})
        kpis_payload.setdefault("avg_speed", canonical_traffic.get("average_speed"))
        kpis_payload.setdefault("avg_wait", kpis_payload.get("avg_wait_min"))
        kpis_payload.setdefault("traffic", canonical_traffic.get("traffic_density"))
        kpis_payload.setdefault("energy", kpis_payload.get("energy_consumed_kwh"))
        return {
            "schema_version": 1,
            "simulation": simulation_payload,
            "kpis": kpis_payload,
            "stations": canonical_stations,
            "vehicles": fleet_vehicles,
            "tracked_vehicles": canonical_vehicles,
            "traffic": canonical_traffic,
            "ppo": ppo_payload,
            "ppo_history": history_payload.get("rewards") and [
                {"ev_id": item.get("vehicle_id"), "selected_station": item.get("station_id"), "action": item.get("action"), "reward": item.get("reward"), "battery": item.get("battery"), "reason": item.get("decision_features")}
                for item in canonical_ppo
            ] or [],
            "summary": summary,
            "metrics": metrics,
            "recommendations": state.get("recommendations") or [],
            "charging_events": state.get("charging_events") or [],
            "latest_recommendation": latest_recommendation,
            "network": network_payload,
            "xai": state.get("xai") or {},
            "validation": validation,
            "tracked_ids": tracked_ids,
            "telemetry": {
                "timestamp": int(time.time() * 1000),
                "step": int(simulation_payload.get("step") or 0),
                "age_ms": telemetry_age_ms,
                "status": telemetry_status,
            },
            "fleet_size": int(self.fleet_size),
            "station_count": int(self.station_count),
            "station_statistics": station_statistics,
            "tracked": int(self.tracked),
            "tracked_ev": canonical_tracked_ev,
            "station": canonical_station,
            "history": canonical_history,
            "charts": charts_payload,
            "stations_summary": stations_payload,
            "vehicles_summary": vehicles_payload,
            "stations": canonical_station_list,
            "vehicles": canonical_vehicle_list,
            "ppo_decision": canonical_ppo_decision,
            "network": network_payload,
        }

    def _build_simulation_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        simulation = dict(state.get("simulation") or {})
        simulation.setdefault("connection_status", "LIVE")
        if "simulation_status" not in simulation:
            raw_status = str(simulation.get("status") or "").strip().lower()
            if raw_status in {"paused", "pause", "stopped"}:
                simulation["simulation_status"] = "PAUSED"
            elif raw_status in {"completed", "finished", "done"}:
                simulation["simulation_status"] = "COMPLETED"
            elif raw_status in {"error", "failed"}:
                simulation["simulation_status"] = "ERROR"
            elif bool(simulation.get("running", True)):
                simulation["simulation_status"] = "RUNNING"
            else:
                simulation["simulation_status"] = "PAUSED"
        if "status" not in simulation:
            simulation["status"] = "play"
        return simulation

    def _build_canonical_simulation(self, state: Dict[str, Any], fallback: Dict[str, Any], vehicles: List[Dict[str, Any]]) -> Dict[str, Any]:
        simulation = state.get("simulation") if isinstance(state.get("simulation"), dict) else {}
        normalized_status = str(simulation.get("status") or simulation.get("simulation_status") or fallback.get("status") or "running").strip().lower()
        if normalized_status in {"play", "run", "running"}:
            status_value = "running"
        elif normalized_status in {"pause", "paused", "stopped"}:
            status_value = "paused"
        elif normalized_status in {"completed", "finished", "done"}:
            status_value = "completed"
        elif normalized_status in {"error", "failed"}:
            status_value = "error"
        else:
            status_value = "running"
        connection_value = simulation.get("connection") or simulation.get("connection_status") or "LIVE"
        step_value = simulation.get("step") if simulation.get("step") is not None else fallback.get("step")
        time_value = simulation.get("time") if simulation.get("time") is not None else fallback.get("time")
        if time_value is None and step_value is not None:
            time_value = step_value
        speed_value = simulation.get("speed") if simulation.get("speed") is not None else fallback.get("speed")
        vehicle_count_value = simulation.get("vehicle_count") if simulation.get("vehicle_count") is not None else simulation.get("ev_count") if simulation.get("ev_count") is not None else fallback.get("vehicle_count") if fallback.get("vehicle_count") is not None else fallback.get("ev_count") if fallback.get("ev_count") is not None else len(vehicles)
        ev_count_value = simulation.get("ev_count") if simulation.get("ev_count") is not None else fallback.get("ev_count") if fallback.get("ev_count") is not None else len(vehicles)
        simulation_status_value = str(simulation.get("simulation_status") or fallback.get("simulation_status") or ("PAUSED" if status_value == "paused" else "RUNNING")).strip().upper()
        if simulation_status_value in {"FINISHED", "DONE"}:
            simulation_status_value = "COMPLETED"
        return {
            "step": self._normalize_value(step_value),
            "time": self._normalize_value(time_value),
            "status": self._normalize_value(status_value),
            "speed": self._normalize_value(speed_value),
            "vehicle_count": self._normalize_value(vehicle_count_value),
            "ev_count": self._normalize_value(ev_count_value),
            "connection": self._normalize_value(connection_value),
            "running": bool(simulation.get("running", fallback.get("running", True))),
            "paused": bool(simulation.get("paused", fallback.get("paused", False))),
            "simulation_status": self._normalize_value(simulation_status_value),
        }

    def _build_canonical_tracked_ev(self, state: Dict[str, Any], fallback: Dict[str, Any], vehicles: List[Dict[str, Any]], ppo_live: Optional[Dict[str, Any]] = None, latest_recommendation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tracked_state = state.get("tracked_ev") if isinstance(state.get("tracked_ev"), dict) else {}
        tracked_vehicle = next((item for item in vehicles if bool(item.get("tracked"))), None)
        if tracked_vehicle is None and vehicles:
            tracked_vehicle = vehicles[0]
        vehicle_id = tracked_state.get("id") or tracked_state.get("vehicle_id") or tracked_vehicle.get("vehicle_id") if tracked_vehicle else None
        x_value = tracked_state.get("x") if tracked_state.get("x") is not None else tracked_vehicle.get("x") if tracked_vehicle else None
        y_value = tracked_state.get("y") if tracked_state.get("y") is not None else tracked_vehicle.get("y") if tracked_vehicle else None
        road_id = tracked_state.get("road_id") or tracked_state.get("current_edge")
        speed_value = tracked_state.get("speed") if tracked_state.get("speed") is not None else tracked_vehicle.get("speed") if tracked_vehicle else None
        battery_value = tracked_state.get("battery") if tracked_state.get("battery") is not None else tracked_state.get("battery_percent") if tracked_state.get("battery_percent") is not None else tracked_vehicle.get("battery") if tracked_vehicle else None
        state_value = tracked_state.get("state") if tracked_state.get("state") is not None else tracked_vehicle.get("state") if tracked_vehicle else None
        selected_station = tracked_state.get("selected_station") if tracked_state.get("selected_station") not in {None, "N/A"} else ppo_live.get("selected_station") if ppo_live else None
        if selected_station in {None, "N/A"} and isinstance(latest_recommendation, dict):
            selected_station = latest_recommendation.get("selected_station")
        destination_value = tracked_state.get("destination") if tracked_state.get("destination") is not None else tracked_vehicle.get("destination") if tracked_vehicle else None
        charging_value = tracked_state.get("charging") if tracked_state.get("charging") is not None else tracked_vehicle.get("charging") if tracked_vehicle else None
        if charging_value is None:
            charging_value = str(state_value or "").lower() in {"charging", "charge"}
        lifecycle_state = tracked_state.get("lifecycle_state") or tracked_state.get("state") or state_value
        state_history = tracked_state.get("state_history") if isinstance(tracked_state.get("state_history"), list) else []
        return {
            "id": self._normalize_value(vehicle_id),
            "x": self._normalize_value(x_value),
            "y": self._normalize_value(y_value),
            "road_id": self._normalize_value(road_id),
            "speed": self._normalize_value(speed_value),
            "battery": self._normalize_value(battery_value),
            "battery_percent": self._normalize_value(battery_value),
            "state": self._normalize_value(lifecycle_state or state_value),
            "lifecycle_state": self._normalize_value(lifecycle_state or state_value),
            "state_history": state_history,
            "selected_station": self._normalize_value(selected_station),
            "destination": self._normalize_value(destination_value),
            "charging": self._normalize_value(charging_value),
            "battery_percent": self._normalize_value(battery_value),
            "soc": self._normalize_value(float(battery_value) / 100.0 if battery_value is not None else None),
        }

    def _build_canonical_ppo_decision(self, state: Dict[str, Any], ppo_live: Optional[Dict[str, Any]], latest_recommendation: Optional[Dict[str, Any]], ppo_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        ppo_state = state.get("ppo_decision") if isinstance(state.get("ppo_decision"), dict) else {}
        if ppo_state:
            ev_value = ppo_state.get("ev") or ppo_state.get("ev_id") or ppo_state.get("vehicle_id")
            station_value = ppo_state.get("station") or ppo_state.get("station_id") or ppo_state.get("selected_station")
            action_value = ppo_state.get("action")
            if action_value is None:
                action_value = ppo_state.get("ppo_action") if ppo_state.get("ppo_action") is not None else station_value
            reward_value = ppo_state.get("reward")
            battery_value = ppo_state.get("battery")
            reason_value = ppo_state.get("reason")
            timestamp_value = ppo_state.get("timestamp")
        elif isinstance(ppo_live, dict):
            ev_value = ppo_live.get("ev_id") or ppo_live.get("vehicle_id")
            station_value = ppo_live.get("selected_station") or ppo_live.get("station_id")
            action_value = ppo_live.get("action")
            if action_value is None:
                action_value = ppo_live.get("ppo_action") if ppo_live.get("ppo_action") is not None else station_value
            reward_value = ppo_live.get("reward")
            battery_value = ppo_live.get("battery")
            reason_value = ppo_live.get("reason")
            timestamp_value = ppo_live.get("decision_time")
        elif isinstance(latest_recommendation, dict):
            ev_value = latest_recommendation.get("vehicle_id")
            station_value = latest_recommendation.get("selected_station")
            action_value = latest_recommendation.get("selected_station")
            reward_value = latest_recommendation.get("ppo_reward")
            battery_value = latest_recommendation.get("battery_pct")
            reason_value = latest_recommendation.get("recommendation_reason") or latest_recommendation.get("reason")
            timestamp_value = latest_recommendation.get("step")
        else:
            ev_value = None
            station_value = None
            action_value = None
            reward_value = None
            battery_value = None
            reason_value = None
            timestamp_value = None
        if not ppo_items and isinstance(ppo_live, dict):
            ppo_items = [ppo_live]
        if not ppo_items and isinstance(latest_recommendation, dict):
            ppo_items = [latest_recommendation]
        if ppo_items and not any(value is not None for value in [ev_value, station_value, action_value, reward_value, battery_value, reason_value, timestamp_value]):
            item = ppo_items[0]
            ev_value = item.get("vehicle_id") or item.get("ev_id") or item.get("ev")
            station_value = item.get("station_id") or item.get("selected_station") or item.get("station")
            action_value = item.get("action")
            if action_value is None:
                action_value = item.get("ppo_action") if item.get("ppo_action") is not None else station_value
            reward_value = item.get("reward") or item.get("ppo_reward")
            battery_value = item.get("battery") or item.get("battery_pct")
            reason_value = item.get("reason") or item.get("decision_features") or item.get("recommendation_reason")
            timestamp_value = item.get("decision_time") or item.get("step") or item.get("timestamp")
        return {
            "ev": self._normalize_value(ev_value),
            "station": self._normalize_value(station_value),
            "action": self._normalize_value(action_value),
            "reward": self._normalize_value(reward_value),
            "battery": self._normalize_value(battery_value),
            "reason": self._normalize_value(reason_value),
            "timestamp": self._normalize_value(timestamp_value),
            "ev_id": self._normalize_value(ev_value),
            "selected_station": self._normalize_value(station_value),
            "station_id": self._normalize_value(station_value),
            "decision_time": self._normalize_value(timestamp_value),
            "vehicle_id": self._normalize_value(ev_value),
        }

    def _build_canonical_station(self, state: Dict[str, Any], fallback: Dict[str, Any], stations: List[Dict[str, Any]], ppo_live: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        station_state = state.get("station") if isinstance(state.get("station"), dict) else {}
        station = next((item for item in stations if str(item.get("station_id") or item.get("id")) == str(station_state.get("id") or station_state.get("station_id") or "")), None) if station_state else None
        if station is None and stations:
            selected_station_id = ppo_live.get("selected_station") if isinstance(ppo_live, dict) else None
            if selected_station_id not in {None, "N/A"}:
                station = next((item for item in stations if str(item.get("station_id") or item.get("id")) == str(selected_station_id)), None)
            if station is None:
                station = stations[0]
        if station is None:
            station = {}
        station_id = station_state.get("id") or station_state.get("station_id") or station.get("station_id") or station.get("id")
        return {
            "id": self._normalize_value(station_id),
            "name": self._normalize_value(station_state.get("name") or station.get("name")),
            "x": self._normalize_value(station_state.get("x") if station_state.get("x") is not None else station.get("x")),
            "y": self._normalize_value(station_state.get("y") if station_state.get("y") is not None else station.get("y")),
            "ports_total": self._normalize_value(station_state.get("ports_total") if station_state.get("ports_total") is not None else station.get("total_ports") or station.get("ports_total")),
            "ports_available": self._normalize_value(station_state.get("ports_available") if station_state.get("ports_available") is not None else station.get("available_ports") or station.get("ports_available")),
            "queue": self._normalize_value(station_state.get("queue") if station_state.get("queue") is not None else station.get("queue") or station.get("queue_length")),
            "price": self._normalize_value(station_state.get("price") if station_state.get("price") is not None else station.get("price") or station.get("price_per_kwh")),
            "wait": self._normalize_value(station_state.get("wait") if station_state.get("wait") is not None else station.get("wait") or station.get("waiting_time_min") or station.get("avg_wait_min") or station.get("avg_wait_estimate")),
            "load": self._normalize_value(station_state.get("load") if station_state.get("load") is not None else station.get("load") or station.get("grid_load_kw") or station.get("grid_load")),
            "utilization": self._normalize_value(station_state.get("utilization") if station_state.get("utilization") is not None else station.get("utilization")),
            "station_id": self._normalize_value(station_id),
            "name": self._normalize_value(station_state.get("name") or station.get("name")),
        }

    def _build_canonical_vehicle_list(self, state: Dict[str, Any], vehicles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        vehicle_payloads = state.get("vehicles") if isinstance(state.get("vehicles"), list) else vehicles
        if not isinstance(vehicle_payloads, list):
            vehicle_payloads = []
        canonical: List[Dict[str, Any]] = []
        for item in vehicle_payloads:
            if not isinstance(item, dict):
                continue
            vehicle_id = item.get("id") or item.get("vehicle_id")
            canonical.append({
                "id": self._normalize_value(vehicle_id),
                "vehicle_id": self._normalize_value(vehicle_id),
                "x": self._normalize_value(item.get("x") if item.get("x") is not None else (item.get("current_position") or {}).get("x")),
                "y": self._normalize_value(item.get("y") if item.get("y") is not None else (item.get("current_position") or {}).get("y")),
                "speed": self._normalize_value(item.get("speed")),
                "battery": self._normalize_value(item.get("battery") if item.get("battery") is not None else item.get("battery_pct") if item.get("battery_pct") is not None else item.get("battery_percent")),
                "battery_percent": self._normalize_value(item.get("battery_percent") if item.get("battery_percent") is not None else item.get("battery") if item.get("battery") is not None else item.get("battery_pct")),
                "state": self._normalize_value(item.get("state") or item.get("status") or item.get("charging_status")),
            })
        if not canonical and vehicles:
            return self._build_canonical_vehicle_list({}, vehicles)
        return canonical

    def _build_canonical_station_list(self, state: Dict[str, Any], stations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        station_payloads = state.get("stations") if isinstance(state.get("stations"), list) else stations
        if not isinstance(station_payloads, list):
            station_payloads = []
        canonical: List[Dict[str, Any]] = []
        for item in station_payloads:
            if not isinstance(item, dict):
                continue
            station_id = item.get("id") or item.get("station_id")
            canonical.append({
                "id": self._normalize_value(station_id),
                "station_id": self._normalize_value(station_id),
                "name": self._normalize_value(item.get("name")),
                "x": self._normalize_value(item.get("x") if item.get("x") is not None else item.get("longitude")),
                "y": self._normalize_value(item.get("y") if item.get("y") is not None else item.get("latitude")),
                "ports_total": self._normalize_value(item.get("ports_total") if item.get("ports_total") is not None else item.get("total_ports")),
                "ports_available": self._normalize_value(item.get("ports_available") if item.get("ports_available") is not None else item.get("available_ports")),
                "queue": self._normalize_value(item.get("queue") if item.get("queue") is not None else item.get("queue_length")),
                "price": self._normalize_value(item.get("price") if item.get("price") is not None else item.get("price_per_kwh")),
                "wait": self._normalize_value(item.get("wait") if item.get("wait") is not None else item.get("wait_minutes") or item.get("avg_wait_min") or item.get("avg_wait_estimate")),
                "load": self._normalize_value(item.get("load") if item.get("load") is not None else item.get("grid_load_kw") or item.get("grid_load")),
                "utilization": self._normalize_value(item.get("utilization")),
            })
        if not canonical and stations:
            return self._build_canonical_station_list({}, stations)
        return canonical

    def _build_canonical_history(self, state: Dict[str, Any], fallback: Dict[str, Any], recommendations: List[Dict[str, Any]], ppo_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        history_state = state.get("history") if isinstance(state.get("history"), dict) else {}
        rewards = history_state.get("rewards") if isinstance(history_state.get("rewards"), list) else fallback.get("rewards") if isinstance(fallback.get("rewards"), list) else []
        recommendations_payload = history_state.get("recommendations") if isinstance(history_state.get("recommendations"), list) else recommendations
        timestamps = history_state.get("timestamps") if isinstance(history_state.get("timestamps"), list) else fallback.get("timestamps") if isinstance(fallback.get("timestamps"), list) else []
        if not rewards and ppo_items:
            rewards = [item.get("reward") or item.get("ppo_reward") for item in ppo_items if item.get("reward") is not None or item.get("ppo_reward") is not None]
        if not timestamps and recommendations_payload:
            timestamps = [index + 1 for index in range(len(recommendations_payload))]
        return {
            "rewards": rewards,
            "recommendations": recommendations_payload,
            "timestamps": timestamps,
        }

    def _canonicalize_stations(self, stations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        canonical: List[Dict[str, Any]] = []
        for station in stations:
            station_id = station.get("station_id")
            name = station.get("name")
            latitude = station.get("latitude")
            longitude = station.get("longitude")
            x = station.get("x")
            y = station.get("y")
            total_ports = station.get("total_ports")
            available_ports = station.get("available_ports")
            occupied_ports = station.get("occupied_ports")
            queue = station.get("queue")
            price_per_kwh = station.get("price_per_kwh")
            waiting_time_min = station.get("waiting_time_min")
            grid_load_kw = station.get("grid_load_kw")
            utilization = station.get("utilization")
            if station.get("lat") is not None and latitude is None:
                latitude = station.get("lat")
            if station.get("lon") is not None and longitude is None:
                longitude = station.get("lon")
            if station.get("queue_length") is not None and queue is None:
                queue = station.get("queue_length")
            if station.get("avg_wait_estimate") is not None and waiting_time_min is None:
                waiting_time_min = station.get("avg_wait_estimate")
            if station.get("avg_wait_min") is not None and waiting_time_min is None:
                waiting_time_min = station.get("avg_wait_min")
            if station.get("price") is not None and price_per_kwh is None:
                price_per_kwh = station.get("price")
            if station.get("price_per_kwh") is not None and price_per_kwh is None:
                price_per_kwh = station.get("price_per_kwh")
            if station.get("grid_load") is not None and grid_load_kw is None:
                grid_load_kw = station.get("grid_load")
            if station.get("occupancy_percent") is not None and utilization is None:
                utilization = station.get("occupancy_percent")
            if latitude is None and station.get("lat") is not None:
                latitude = station.get("lat")
            if longitude is None and station.get("lon") is not None:
                longitude = station.get("lon")
            if x is None and station.get("x") is not None:
                x = station.get("x")
            if y is None and station.get("y") is not None:
                y = station.get("y")
            if station_id is None:
                station_id = f"station_{len(canonical) + 1}"
            if name is None:
                name = str(station_id)
            if latitude is None:
                latitude = 0.0
            if longitude is None:
                longitude = 0.0
            if x is None:
                x = 0.0
            if y is None:
                y = 0.0
            if total_ports is None:
                total_ports = 0
            if available_ports is None:
                available_ports = 0
            if occupied_ports is None:
                occupied_ports = 0
            if queue is None:
                queue = 0
            if price_per_kwh is None:
                price_per_kwh = 0.0
            if waiting_time_min is None:
                waiting_time_min = 0.0
            if grid_load_kw is None:
                grid_load_kw = 0.0
            if utilization is None and total_ports not in {None, 0}:
                utilization = (float(occupied_ports) / float(total_ports)) if occupied_ports is not None else 0.0
            if utilization is None:
                utilization = 0.0
            try:
                utilization_value = float(utilization) if utilization is not None else 0.0
            except (TypeError, ValueError):
                utilization_value = 0.0
            queue_length = queue if queue is not None else max(0, int(total_ports or 0) - int(available_ports or 0))
            canonical.append({
                "id": self._normalize_value(station_id),
                "station_id": self._normalize_value(station_id),
                "name": self._normalize_value(name),
                "latitude": self._normalize_value(latitude),
                "longitude": self._normalize_value(longitude),
                "x": self._normalize_value(x),
                "y": self._normalize_value(y),
                "total_ports": self._normalize_value(total_ports),
                "available_ports": self._normalize_value(available_ports),
                "occupied_ports": self._normalize_value(occupied_ports),
                "queue": self._normalize_value(queue_length),
                "queue_length": self._normalize_value(queue_length),
                "price_per_kwh": self._normalize_value(price_per_kwh),
                "waiting_time_min": self._normalize_value(waiting_time_min),
                "grid_load_kw": self._normalize_value(grid_load_kw),
                "load_kw": self._normalize_value(grid_load_kw),
                "utilization": self._normalize_value(utilization_value),
                "status": self._normalize_value("charging" if occupied_ports and int(occupied_ports) > 0 else "idle"),
            })
        return canonical

    def _canonicalize_vehicles(self, vehicles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        canonical: List[Dict[str, Any]] = []
        for vehicle in vehicles:
            vehicle_id = vehicle.get("vehicle_id") or vehicle.get("id")
            state_value = vehicle.get("state") or vehicle.get("status") or vehicle.get("charging_status")
            current_station = vehicle.get("current_station") or vehicle.get("recommended_station") or vehicle.get("station_id")
            recommended_station = vehicle.get("recommended_station") or vehicle.get("station_id")
            recommendation_score = vehicle.get("recommendation_score")
            price = vehicle.get("price")
            queue = vehicle.get("queue")
            waiting_time_min = vehicle.get("waiting_time_min")
            available_ports = vehicle.get("available_ports")
            grid_load_kw = vehicle.get("grid_load_kw")
            battery = vehicle.get("battery")
            if battery is None and vehicle.get("battery_pct") is not None:
                battery = vehicle.get("battery_pct")
            if price is None and vehicle.get("charging_price_per_kwh") is not None:
                price = vehicle.get("charging_price_per_kwh")
            if queue is None and vehicle.get("queue_length") is not None:
                queue = vehicle.get("queue_length")
            if waiting_time_min is None and vehicle.get("waiting_time") is not None:
                waiting_time_min = vehicle.get("waiting_time")
            if available_ports is None and vehicle.get("available_ports") is not None:
                available_ports = vehicle.get("available_ports")
            if grid_load_kw is None and vehicle.get("grid_load") is not None:
                grid_load_kw = vehicle.get("grid_load")
            if recommendation_score is None:
                recommendation_score = 0.0
            current_position = vehicle.get("current_position") or {}
            x_value = vehicle.get("x") if vehicle.get("x") is not None else current_position.get("x")
            y_value = vehicle.get("y") if vehicle.get("y") is not None else current_position.get("y")
            if vehicle_id is None:
                vehicle_id = f"vehicle_{len(canonical) + 1}"
            soc = None
            battery_value = battery
            if battery_value is None and vehicle.get("battery_pct") is not None:
                battery_value = vehicle.get("battery_pct")
            if battery_value is not None:
                try:
                    soc = float(battery_value) / 100.0 if float(battery_value) >= 0 else None
                except (TypeError, ValueError):
                    soc = None
            if state_value is None:
                state_value = "unknown"
            if current_station is None:
                current_station = "N/A"
            if recommended_station is None:
                recommended_station = "N/A"
            if x_value is None:
                x_value = 0.0
            if y_value is None:
                y_value = 0.0
            if battery is None:
                battery = 0.0
            if price is None:
                price = 0.0
            if queue is None:
                queue = 0
            if waiting_time_min is None:
                waiting_time_min = 0.0
            if available_ports is None:
                available_ports = 0
            if grid_load_kw is None:
                grid_load_kw = 0.0
            if vehicle.get("speed") is None:
                vehicle_speed = 0.0
            else:
                vehicle_speed = vehicle.get("speed")
            canonical.append({
                "vehicle_id": self._normalize_value(vehicle_id),
                "id": self._normalize_value(vehicle_id),
                "x": self._normalize_value(x_value),
                "y": self._normalize_value(y_value),
                "speed": self._normalize_value(vehicle_speed),
                "battery": self._normalize_value(battery),
                "soc": self._normalize_value(soc),
                "battery_percent": self._normalize_value(battery),
                "state": self._normalize_value(state_value),
                "current_station": self._normalize_value(current_station),
                "recommended_station": self._normalize_value(recommended_station),
                "recommendation_score": self._normalize_value(recommendation_score),
                "reward": self._normalize_value(vehicle.get("reward") if vehicle.get("reward") is not None else vehicle.get("ppo_reward") if vehicle.get("ppo_reward") is not None else 0.0),
                "current_edge": self._normalize_value(vehicle.get("current_edge")),
                "destination": self._normalize_value(vehicle.get("destination") or vehicle.get("destination_edge")),
                "tracked": False,
                "queue": self._normalize_value(queue),
                "price": self._normalize_value(price),
                "waiting_time_min": self._normalize_value(waiting_time_min),
                "available_ports": self._normalize_value(available_ports),
                "grid_load_kw": self._normalize_value(grid_load_kw),
            })
        return canonical

    def _canonicalize_traffic(self, traffic_items: List[Dict[str, Any]], simulation: Optional[Dict[str, Any]], kpis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        traffic = {"average_speed": None, "traffic_density": None, "congestion_percent": None, "vehicle_count": None}
        if isinstance(kpis, dict):
            traffic["average_speed"] = kpis.get("avg_speed") if kpis.get("avg_speed") is not None else kpis.get("average_speed")
            traffic["traffic_density"] = kpis.get("traffic_density")
        if isinstance(simulation, dict):
            traffic["congestion_percent"] = simulation.get("congestion_percent")
            traffic["vehicle_count"] = simulation.get("ev_count") or simulation.get("fleet_size")
        if traffic_items:
            first = traffic_items[0]
            if traffic.get("average_speed") is None:
                traffic["average_speed"] = first.get("speed")
            if traffic.get("traffic_density") is None:
                traffic["traffic_density"] = first.get("density")
            if traffic.get("congestion_percent") is None:
                traffic["congestion_percent"] = first.get("congestion")
            if traffic.get("vehicle_count") is None:
                traffic["vehicle_count"] = len(traffic_items)
        if isinstance(traffic.get("congestion_percent"), (int, float)) and traffic.get("congestion_percent") is not None and traffic.get("congestion_percent") <= 1.0:
            traffic["congestion_percent"] = float(traffic["congestion_percent"]) * 100.0
        return {
            "average_speed": self._normalize_value(traffic.get("average_speed")),
            "traffic_density": self._normalize_value(traffic.get("traffic_density")),
            "congestion_percent": self._normalize_value(traffic.get("congestion_percent")),
            "vehicle_count": self._normalize_value(traffic.get("vehicle_count")),
        }

    def _canonicalize_ppo(self, ppo_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        canonical: List[Dict[str, Any]] = []
        for item in ppo_items:
            battery = item.get("battery") or item.get("battery_pct")
            try:
                soc = float(battery) / 100.0 if battery is not None else None
            except (TypeError, ValueError):
                soc = None
            canonical.append({
                "vehicle_id": self._normalize_value(item.get("vehicle_id") or item.get("ev_id")),
                "action": self._normalize_value(item.get("action")),
                "station_id": self._normalize_value(item.get("station_id") or item.get("selected_station")),
                "reward": self._normalize_value(item.get("reward") or item.get("ppo_reward")),
                "battery": self._normalize_value(battery),
                "soc": self._normalize_value(soc),
                "decision_time": self._normalize_value(item.get("decision_time") or item.get("step")),
                "decision_features": self._normalize_value(item.get("decision_features") or item.get("reason")),
                "reason": self._normalize_value(item.get("reason") or item.get("decision_features")),
                "distance": self._normalize_value(item.get("distance") or item.get("travel_distance")),
                "wait": self._normalize_value(item.get("wait") or item.get("waiting_time")),
                "price": self._normalize_value(item.get("price") or item.get("charging_price_per_kwh")),
                "load": self._normalize_value(item.get("load") or item.get("grid_load") or item.get("grid_load_kw")),
                "available_ports": self._normalize_value(item.get("available_ports") or item.get("free_ports")),
            })
        return canonical

    def _normalize_value(self, value: Any) -> Any:
        return value if value is not None else "N/A"

    def _coerce_float(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped in {"", "N/A", "None", "null", "nan", "NaN"}:
                return None
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(coerced):
            return None
        return coerced

    def _coerce_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped in {"", "N/A", "None", "null"}:
                return None
            try:
                return int(float(stripped))
            except ValueError:
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _derive_traffic_level(self, congestion_percent: Any) -> str:
        try:
            value = float(congestion_percent)
        except (TypeError, ValueError):
            return "N/A"
        if value < 20.0:
            return "low"
        if value < 50.0:
            return "medium"
        return "high"

    def _build_latest_ppo(self, ppo_items: List[Dict[str, Any]], latest_recommendation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if ppo_items:
            item = ppo_items[0]
            battery = item.get("battery") or item.get("battery_pct")
            try:
                soc = float(battery) / 100.0 if battery is not None else None
            except (TypeError, ValueError):
                soc = None
            return {
                "timestamp": self._normalize_value(item.get("decision_time") or item.get("step") or item.get("timestamp")),
                "ev_id": self._normalize_value(item.get("vehicle_id") or item.get("ev_id")),
                "selected_station": self._normalize_value(item.get("station_id") or item.get("selected_station")),
                "action": self._normalize_value(item.get("action")),
                "reward": self._normalize_value(item.get("reward") or item.get("ppo_reward")),
                "battery": self._normalize_value(battery),
                "soc": self._normalize_value(soc),
                "reason": self._normalize_value(item.get("reason") or item.get("decision_features") or item.get("recommendation_reason")),
                "distance": self._normalize_value(item.get("distance") or item.get("travel_distance")),
                "wait": self._normalize_value(item.get("wait") or item.get("waiting_time")),
                "price": self._normalize_value(item.get("price") or item.get("charging_price_per_kwh")),
                "load": self._normalize_value(item.get("load") or item.get("grid_load") or item.get("grid_load_kw")),
                "available_ports": self._normalize_value(item.get("available_ports") or item.get("free_ports")),
            }
        if isinstance(latest_recommendation, dict):
            reason = latest_recommendation.get("recommendation_reason") or latest_recommendation.get("reason") or latest_recommendation.get("ppo_observation")
            battery = latest_recommendation.get("battery_pct")
            try:
                soc = float(battery) / 100.0 if battery is not None else None
            except (TypeError, ValueError):
                soc = None
            return {
                "timestamp": self._normalize_value(latest_recommendation.get("step") or latest_recommendation.get("decision_time") or latest_recommendation.get("timestamp")),
                "ev_id": self._normalize_value(latest_recommendation.get("vehicle_id")),
                "selected_station": self._normalize_value(latest_recommendation.get("selected_station")),
                "action": self._normalize_value(latest_recommendation.get("ppo_action") or latest_recommendation.get("action") or latest_recommendation.get("selected_station")),
                "reward": self._normalize_value(latest_recommendation.get("ppo_reward")),
                "battery": self._normalize_value(battery),
                "soc": self._normalize_value(soc),
                "reason": self._normalize_value(reason),
                "distance": self._normalize_value(latest_recommendation.get("travel_distance")),
                "wait": self._normalize_value(latest_recommendation.get("waiting_time")),
                "price": self._normalize_value(latest_recommendation.get("charging_price_per_kwh")),
                "load": self._normalize_value(latest_recommendation.get("grid_load")),
                "available_ports": self._normalize_value(latest_recommendation.get("available_ports")),
            }
        return {"ev_id": "N/A", "selected_station": "N/A", "action": "N/A", "reward": "N/A", "battery": "N/A", "reason": "N/A"}

    def _build_history_payload(self, ppo_items: List[Dict[str, Any]], recommendations: List[Dict[str, Any]], stations: List[Dict[str, Any]]) -> Dict[str, Any]:
        rewards: list[float] = []
        timestamps: list[int] = []

        history_items: list[Dict[str, Any]] = list(recommendations)
        if not history_items:
            history_items = list(ppo_items)

        for item in history_items:
            reward = self._coerce_float(item.get("reward"))
            if reward is None:
                reward = self._coerce_float(item.get("ppo_reward"))
            if reward is not None:
                rewards.append(reward)
                timestamp = self._coerce_int(item.get("decision_time") or item.get("step"))
                if timestamp is None:
                    timestamp = len(timestamps) + 1
                timestamps.append(timestamp)

        if not rewards and ppo_items:
            for item in ppo_items:
                reward = self._coerce_float(item.get("reward"))
                if reward is not None:
                    rewards.append(reward)
                    timestamp = self._coerce_int(item.get("decision_time") or len(timestamps) + 1)
                    if timestamp is None:
                        timestamp = len(timestamps) + 1
                    timestamps.append(timestamp)

        utilization = []
        for station in stations:
            value = self._coerce_float(station.get("utilization"))
            if value is not None:
                utilization.append(value)

        if not timestamps:
            timestamps = [int(index + 1) for index in range(len(rewards))]

        return {
            "rewards": rewards,
            "utilization": utilization,
            "timestamps": timestamps,
        }

    def _build_charts_payload(self, ppo_items: List[Dict[str, Any]], recommendations: List[Dict[str, Any]], stations: List[Dict[str, Any]]) -> Dict[str, Any]:
        reward_series: list[Dict[str, Any]] = []
        history_items: list[Dict[str, Any]] = list(ppo_items)
        if not history_items:
            history_items = list(recommendations)
        else:
            history_items.extend(recommendations)

        for item in history_items:
            reward = self._coerce_float(item.get("reward"))
            if reward is None:
                reward = self._coerce_float(item.get("ppo_reward"))
            if reward is None:
                continue
            reward_series.append({
                "ev_id": self._normalize_value(item.get("vehicle_id") or item.get("ev_id")),
                "station_id": self._normalize_value(item.get("station_id") or item.get("selected_station")),
                "reward": reward,
                "timestamp": self._coerce_int(item.get("decision_time") or item.get("step")) or len(reward_series) + 1,
            })

        station_series: list[Dict[str, Any]] = []
        for station in stations:
            station_id = station.get("station_id") or station.get("id")
            available_ports = self._coerce_int(station.get("available_ports"))
            total_ports = self._coerce_int(station.get("total_ports"))
            queue = self._coerce_int(station.get("queue"))
            utilization = self._coerce_float(station.get("utilization"))
            if station_id is None and available_ports is None and total_ports is None and queue is None and utilization is None:
                continue
            station_series.append({
                "id": self._normalize_value(station_id),
                "name": self._normalize_value(station.get("name")),
                "available_ports": available_ports,
                "total_ports": total_ports,
                "queue": queue,
                "utilization": utilization,
                "price": self._coerce_float(station.get("price_per_kwh")),
                "wait_minutes": self._coerce_float(station.get("waiting_time_min")),
                "load_kw": self._coerce_float(station.get("grid_load_kw")),
            })

        return {
            "reward_series": reward_series,
            "station_series": station_series,
        }

    def _build_station_payload(self, stations: List[Dict[str, Any]], ppo_live: Dict[str, Any]) -> Dict[str, Any]:
        if not stations:
            return {"id": "N/A", "name": "N/A", "latitude": None, "longitude": None, "ports_total": None, "ports_available": None, "queue": None, "price": None, "wait_minutes": None, "load_kw": None, "utilization": None}
        station = stations[0]
        selected_station_id = ppo_live.get("selected_station") if isinstance(ppo_live, dict) else None
        if selected_station_id not in {None, "N/A"}:
            candidate = next((item for item in stations if str(item.get("station_id") or item.get("id")) == str(selected_station_id)), None)
            if candidate is not None:
                station = candidate
        return {
            "id": self._normalize_value(station.get("station_id") or station.get("id")),
            "name": self._normalize_value(station.get("name")),
            "latitude": self._normalize_value(station.get("latitude")),
            "longitude": self._normalize_value(station.get("longitude")),
            "ports_total": self._normalize_value(station.get("total_ports")),
            "ports_available": self._normalize_value(station.get("available_ports")),
            "queue": self._normalize_value(station.get("queue")),
            "price": self._normalize_value(station.get("price_per_kwh")),
            "wait_minutes": self._normalize_value(station.get("waiting_time_min")),
            "load_kw": self._normalize_value(station.get("grid_load_kw")),
            "utilization": self._normalize_value(station.get("utilization")),
        }

    def _build_station_statistics(self, stations: List[Dict[str, Any]]) -> Dict[str, int]:
        registered = len(stations)
        real_osm = 0
        active = 0
        for station in stations:
            station_id = str(station.get("station_id") or station.get("id") or "").lower()
            data_source = str(station.get("data_source") or "").lower()
            if station_id.startswith("osm_") or "osm" in data_source or data_source == "openstreetmap":
                real_osm += 1
            occupied = self._coerce_int(station.get("occupied_ports")) or 0
            queue = self._coerce_int(station.get("queue") or station.get("queue_length")) or 0
            charging_state = str(station.get("charging_state") or station.get("status") or "").lower()
            if occupied > 0 or queue > 0 or charging_state in {"active", "charging", "queued"}:
                active += 1
        return {"real_osm": real_osm, "registered": registered, "active": active}

    def _build_tracked_ev_payload(self, vehicles: List[Dict[str, Any]], ppo_live: Optional[Dict[str, Any]] = None, latest_recommendation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tracked_vehicle = next((item for item in vehicles if bool(item.get("tracked"))), None)
        if tracked_vehicle is None and vehicles:
            tracked_vehicle = vehicles[0]
        if tracked_vehicle is None:
            return {"id": "N/A", "x": None, "y": None, "speed": None, "battery_percent": None, "soc": None, "state": "N/A", "current_edge": None, "destination": None, "selected_station": None}
        selected_station = None
        if isinstance(ppo_live, dict):
            selected_station = ppo_live.get("selected_station")
        if selected_station in {None, "N/A"} and isinstance(latest_recommendation, dict):
            selected_station = latest_recommendation.get("selected_station")
        if selected_station in {None, "N/A"}:
            selected_station = tracked_vehicle.get("recommended_station")
        battery_value = tracked_vehicle.get("battery")
        try:
            soc = float(battery_value) / 100.0 if battery_value is not None else None
        except (TypeError, ValueError):
            soc = None
        return {
            "id": self._normalize_value(tracked_vehicle.get("vehicle_id") or tracked_vehicle.get("id")),
            "x": self._normalize_value(tracked_vehicle.get("x")),
            "y": self._normalize_value(tracked_vehicle.get("y")),
            "speed": self._normalize_value(tracked_vehicle.get("speed")),
            "battery_percent": self._normalize_value(tracked_vehicle.get("battery")),
            "soc": self._normalize_value(soc),
            "state": self._normalize_value(tracked_vehicle.get("state")),
            "current_edge": self._normalize_value(tracked_vehicle.get("current_edge")),
            "destination": self._normalize_value(tracked_vehicle.get("destination")),
            "selected_station": self._normalize_value(selected_station),
            "charging_status": self._normalize_value(tracked_vehicle.get("state")),
            "waiting_time": self._normalize_value(tracked_vehicle.get("waiting_time_min")),
            "simulation_time": self._normalize_value(None),
        }

    def _build_station_summary_payload(self, stations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "id": self._normalize_value(station.get("station_id") or station.get("id")),
                "x": self._normalize_value(station.get("x")),
                "y": self._normalize_value(station.get("y")),
                "ports_total": self._normalize_value(station.get("total_ports")),
                "ports_available": self._normalize_value(station.get("available_ports")),
                "queue": self._normalize_value(station.get("queue")),
                "price": self._normalize_value(station.get("price_per_kwh")),
                "load_kw": self._normalize_value(station.get("grid_load_kw")),
                "utilization": self._normalize_value(station.get("utilization")),
            }
            for station in stations
        ]

    def _build_vehicle_summary_payload(self, vehicles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "id": self._normalize_value(vehicle.get("vehicle_id") or vehicle.get("id")),
                "x": self._normalize_value(vehicle.get("x")),
                "y": self._normalize_value(vehicle.get("y")),
                "speed": self._normalize_value(vehicle.get("speed")),
                "battery": self._normalize_value(vehicle.get("battery")),
                "tracked": bool(vehicle.get("tracked")),
            }
            for vehicle in vehicles
        ]

    def _validate_dashboard_payload(self, stations: List[Dict[str, Any]], vehicles: List[Dict[str, Any]], traffic: Dict[str, Any], ppo: List[Dict[str, Any]]) -> Dict[str, Any]:
        missing_fields: List[str] = []
        for station in stations:
            for field in REQUIRED_STATION_FIELDS:
                if field not in station or station.get(field) is None or station.get(field) == "N/A":
                    missing_fields.append(f"station.{field}")
        for vehicle in vehicles:
            for field in REQUIRED_VEHICLE_FIELDS:
                if field not in vehicle or vehicle.get(field) is None or vehicle.get(field) == "N/A":
                    missing_fields.append(f"vehicle.{field}")
        for field in REQUIRED_TRAFFIC_FIELDS:
            if field not in traffic or traffic.get(field) is None or traffic.get(field) == "N/A":
                missing_fields.append(f"traffic.{field}")
        for item in ppo:
            for field in REQUIRED_PPO_FIELDS:
                if field not in item or item.get(field) is None or item.get(field) == "N/A":
                    missing_fields.append(f"ppo.{field}")
        return {"missing_fields": missing_fields}

    def finalize(self, state: Optional[Dict[str, Any]] = None) -> None:
        final_state = state or self._last_state or self._sample_state()
        canonical_state = self._build_canonical_state(final_state if isinstance(final_state, dict) else {})
        self._write_state(canonical_state)
        self._write_history(canonical_state)
        self._write_research_dashboard(canonical_state)
        self._write_summary(canonical_state.get("summary") or {})

    def _sample_state(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "simulation": {"connection_status": "LIVE", "simulation_status": "RUNNING"},
            "kpis": {},
            "stations": [],
            "tracked_vehicles": [],
            "traffic": {
                "average_speed": "N/A",
                "traffic_density": "N/A",
                "congestion_percent": "N/A",
                "vehicle_count": "N/A",
            },
            "ppo": [],
            "summary": {"charging_vehicles": 0, "avg_wait_min": "N/A", "active_assignments": 0, "total_vehicles": 0},
            "metrics": [],
            "recommendations": [],
            "charging_events": [],
            "latest_recommendation": None,
            "network": {},
            "xai": {},
            "validation": {"missing_fields": ["station.station_id", "vehicle.vehicle_id", "traffic.average_speed"]},
            "tracked_ids": [],
        }

    def _build_summary(self, vehicles: List[Dict[str, Any]], stations: List[Dict[str, Any]], metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        charging_vehicles = sum(1 for vehicle in vehicles if str(vehicle.get("charging_status") or "").lower() == "charging")
        wait_values = [float(station.get("avg_wait_min", 0.0)) for station in stations if station.get("avg_wait_min") is not None]
        avg_wait_min = round(sum(wait_values) / len(wait_values), 2) if wait_values else 0.0
        latest_metric = metrics[-1] if metrics else {}
        return {
            "charging_vehicles": charging_vehicles,
            "avg_wait_min": avg_wait_min,
            "active_assignments": int(latest_metric.get("assigned", 0)),
            "total_vehicles": int(latest_metric.get("total_vehicles", len(vehicles))),
        }

    def _write_summary(self, summary: Dict[str, Any]) -> None:
        self._atomic_write_json(self.summary_path, summary)

    def _format_dashboard_value(self, value: Any, fallback: str = "N/A") -> str:
        if value is None:
            return fallback
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 0:
                return "0"
            if value != value:
                return fallback
            return f"{value:.2f}".rstrip("0").rstrip(".") if "." in f"{value:.2f}" else str(int(value))
        if isinstance(value, str):
            return value if value.strip() else fallback
        return str(value)

    def _build_digital_twin_html(self, state: Dict[str, Any]) -> str:
        simulation = state.get("simulation") or {}
        kpis = state.get("kpis") or {}
        stations_raw = state.get("stations") or []
        stations = [item for item in stations_raw if isinstance(item, dict)] if isinstance(stations_raw, list) else []
        vehicles = [item for item in (state.get("tracked_vehicles") or []) if isinstance(item, dict)] if isinstance(state.get("tracked_vehicles"), list) else []
        traffic = state.get("traffic") or {}
        ppo_items = state.get("ppo") or []
        if isinstance(ppo_items, dict):
            ppo_items = [ppo_items]
        elif not isinstance(ppo_items, list):
            ppo_items = []
        ppo_items = [item for item in ppo_items if isinstance(item, dict)]
        latest_recommendation = state.get("latest_recommendation") if isinstance(state.get("latest_recommendation"), dict) else None
        latest_ppo = ppo_items[0] if ppo_items else None
        if latest_ppo is None and latest_recommendation is not None:
            latest_ppo = {
                "vehicle_id": latest_recommendation.get("vehicle_id"),
                "action": latest_recommendation.get("selected_station"),
                "station_id": latest_recommendation.get("selected_station"),
                "reward": latest_recommendation.get("ppo_reward"),
                "battery": latest_recommendation.get("battery_pct"),
                "decision_time": latest_recommendation.get("step"),
                "decision_features": latest_recommendation.get("recommendation_reason") or latest_recommendation.get("reason") or latest_recommendation.get("ppo_observation"),
            }
        if latest_ppo is not None:
            latest_ppo = {
                **latest_ppo,
                "reason": latest_ppo.get("reason") or latest_ppo.get("decision_features") or latest_ppo.get("recommendation_reason"),
                "decision_features": latest_ppo.get("decision_features") or latest_ppo.get("reason") or latest_ppo.get("recommendation_reason"),
            }
        selected_station_id = None
        if latest_ppo and latest_ppo.get("station_id"):
            selected_station_id = latest_ppo.get("station_id")
        elif vehicles:
            selected_station_id = vehicles[0].get("recommended_station")
        selected_station = None
        if selected_station_id:
            selected_station = next((station for station in stations if str(station.get("station_id")) == str(selected_station_id)), None)
        if selected_station is None and stations:
            selected_station = stations[0]
        summary = state.get("summary") or {}

        template = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EV DIGITAL TWIN</title>
  <style>
    :root { color-scheme: dark; --bg:#07111d; --panel:#101b2b; --panel-2:#16253a; --line:#22364f; --accent:#3dd9c7; --accent-2:#6ea8ff; --warn:#ffb84d; --text:#e5f0ff; --muted:#8ea8c3; }
    * { box-sizing: border-box; }
    body { margin:0; background:linear-gradient(135deg, var(--bg), #0d1b29); color:var(--text); font-family: 'Segoe UI', Roboto, Arial, sans-serif; }
    .shell { max-width: 1500px; margin:0 auto; padding:24px; display:flex; flex-direction:column; gap:20px; }
    .title { font-size: clamp(1.6rem, 2.4vw, 2.3rem); font-weight:700; letter-spacing:0.2em; margin:0; text-transform:uppercase; }
    .subtle { color:var(--muted); margin-top:4px; font-size:0.95rem; }
    .grid { display:grid; gap:20px; }
    .top-grid { grid-template-columns: 2fr 1fr; }
    .mid-grid { grid-template-columns: 1fr 1fr 1fr; }
    .bottom-grid { grid-template-columns: 1fr 1fr; }
    .panel { background:rgba(16,27,43,0.95); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 12px 30px rgba(0,0,0,0.25); }
    .panel h2 { font-size:1rem; margin:0 0 10px; color:var(--accent); text-transform:uppercase; letter-spacing:0.12em; }
    .metrics { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; }
    .metric { background:var(--panel-2); border:1px solid var(--line); border-radius:12px; padding:12px; }
    .metric .label { font-size:0.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.12em; }
    .metric .value { font-size:1.1rem; font-weight:600; margin-top:6px; }
    .network { position:relative; min-height:320px; }
    svg { width:100%; height:100%; min-height:280px; }
    .legend { display:flex; gap:12px; font-size:0.82rem; color:var(--muted); margin-top:8px; }
    .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
    button { border:none; border-radius:999px; padding:10px 14px; background:var(--accent-2); color:white; font-weight:600; cursor:pointer; }
    .button-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
    .pill { background:var(--panel-2); padding:8px 10px; border-radius:999px; border:1px solid var(--line); color:var(--muted); font-size:0.9rem; }
    .station-card { display:grid; gap:8px; }
    .station-card .big { font-size:1.6rem; font-weight:700; }
    .value-list { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    .value-list .item { background:var(--panel-2); border-radius:12px; padding:10px; border:1px solid var(--line); }
    .value-list .item .label { color:var(--muted); font-size:0.73rem; text-transform:uppercase; }
    .value-list .item .value { margin-top:4px; font-weight:600; }
    .chart { height:220px; }
    .bar-row { display:flex; align-items:flex-end; gap:10px; height:160px; margin-top:10px; }
    .bar { flex:1; background:linear-gradient(180deg, var(--accent-2), var(--accent)); border-radius:8px 8px 2px 2px; min-height:6px; position:relative; }
    .bar span { position:absolute; bottom:-24px; width:100%; text-align:center; color:var(--muted); font-size:0.75rem; }
    .small { color:var(--muted); font-size:0.9rem; }
    @media (max-width: 1000px) { .top-grid, .mid-grid, .bottom-grid { grid-template-columns:1fr !important; } }
  </style>
</head>
<body>
  <div class="shell">
    <div class="panel">
      <h1 class="title">EV DIGITAL TWIN</h1>
      <div class="subtle">EV Charging Research Dashboard • Live telemetry from the current simulation run • updated directly from dashboard_state.json</div>
    </div>

    <div class="grid top-grid" style="display:grid; grid-template-columns:2fr 1fr;">
      <section class="panel network">
        <h2>Bengaluru Network</h2>
        <svg id="network-svg" viewBox="0 0 680 380" role="img" aria-label="Network view"></svg>
        <div class="legend">
          <span><span class="dot" style="background:#6ea8ff"></span>Station</span>
          <span><span class="dot" style="background:#3dd9c7"></span>Tracked EV</span>
          <span><span class="dot" style="background:#ffb84d"></span>Selected Station</span>
          <span><span class="dot" style="background:#7fd3ff"></span>Other EV</span>
        </div>
      </section>
      <section class="panel">
        <h2>Simulation</h2>
        <div class="button-row">
          <button data-action="play">▶ Play</button>
          <button data-action="pause">⏸ Pause</button>
          <button data-action="reset">↻ Reset</button>
          <label class="pill" style="display:inline-flex;align-items:center;gap:6px;">
            Speed
            <select id="speed-select" style="background:transparent;color:inherit;border:none;outline:none;">
              <option value="0.5">0.5x</option>
              <option value="1.0" selected>1.0x</option>
              <option value="2.0">2.0x</option>
              <option value="5.0">5.0x</option>
            </select>
          </label>
        </div>
        <div class="value-list" style="margin-top:14px;">
          <div class="item"><div class="label">Speed</div><div class="value" id="sim-speed">__SIM_SPEED__</div></div>
          <div class="item"><div class="label">Time</div><div class="value" id="sim-time">__SIM_TIME__</div></div>
          <div class="item"><div class="label">Vehicles</div><div class="value" id="sim-vehicles">__SIM_VEHICLES__</div></div>
          <div class="item"><div class="label">EVs</div><div class="value" id="sim-evs">__SIM_EVS__</div></div>
          <div class="item"><div class="label">Congestion</div><div class="value" id="sim-congestion">__SIM_CONGESTION__</div></div>
          <div class="item"><div class="label">Step</div><div class="value" id="sim-step">__SIM_STEP__</div></div>
          <div class="item"><div class="label">Connection</div><div class="value" id="sim-connection">__SIM_CONNECTION__</div></div>
          <div class="item"><div class="label">Status</div><div class="value" id="sim-status">__SIM_STATUS__</div></div>
          <div class="item"><div class="label">Recommendations</div><div class="value" id="stat-recs">0</div></div>
          <div class="item"><div class="label">Charging Started</div><div class="value" id="stat-charging-started">0</div></div>
          <div class="item"><div class="label">Charging Done</div><div class="value" id="stat-charging-done">0</div></div>
          <div class="item"><div class="label">Queue Events</div><div class="value" id="stat-queue-events">0</div></div>
        </div>
      </section>
    </div>

    <div class="grid mid-grid" style="display:grid; grid-template-columns:1fr 1fr 1fr;">
      <section class="panel">
        <h2>Live KPIs</h2>
        <div class="metrics">
          <div class="metric"><div class="label">Avg Speed</div><div class="value" id="kpi-speed">__KPI_SPEED__</div></div>
          <div class="metric"><div class="label">Traffic</div><div class="value" id="kpi-traffic">__KPI_TRAFFIC__</div></div>
          <div class="metric"><div class="label">Avg Wait</div><div class="value" id="kpi-wait">__KPI_WAIT__</div></div>
          <div class="metric"><div class="label">Energy</div><div class="value" id="kpi-energy">__KPI_ENERGY__</div></div>
        </div>
      </section>
      <section class="panel station-card">
        <h2>EV Station</h2>
        <div class="big" id="station-name">__STATION_NAME__</div>
        <div class="small" id="station-id">__STATION_ID__</div>
        <div class="value-list">
          <div class="item"><div class="label">Ports</div><div class="value" id="station-ports">__STATION_PORTS__</div></div>
          <div class="item"><div class="label">Queue</div><div class="value" id="station-queue">__STATION_QUEUE__</div></div>
          <div class="item"><div class="label">Price</div><div class="value" id="station-price">__STATION_PRICE__</div></div>
          <div class="item"><div class="label">Wait</div><div class="value" id="station-wait">__STATION_WAIT__</div></div>
          <div class="item"><div class="label">Load</div><div class="value" id="station-load">__STATION_LOAD__</div></div>
          <div class="item"><div class="label">Utilization</div><div class="value" id="station-util">__STATION_UTIL__</div></div>
        </div>
      </section>
      <section class="panel">
        <h2>PPO Decision</h2>
        <div class="value-list">
          <div class="item"><div class="label">EV</div><div class="value" id="ppo-vehicle">__PPO_VEHICLE__</div></div>
          <div class="item"><div class="label">Action</div><div class="value" id="ppo-action">__PPO_ACTION__</div></div>
          <div class="item"><div class="label">Station</div><div class="value" id="ppo-station">__PPO_STATION__</div></div>
          <div class="item"><div class="label">Reward</div><div class="value" id="ppo-reward">__PPO_REWARD__</div></div>
          <div class="item"><div class="label">Battery</div><div class="value" id="ppo-battery">__PPO_BATTERY__</div></div>
          <div class="item"><div class="label">Reason</div><div class="value" id="ppo-reason">__PPO_REASON__</div></div>
        </div>
      </section>
    </div>

    <div class="grid bottom-grid" style="display:grid; grid-template-columns:1fr 1fr;">
      <section class="panel">
        <h2>PPO Reward Trend</h2>
        <div class="chart">
          <svg id="reward-chart" viewBox="0 0 340 180" aria-label="Reward trend"></svg>
        </div>
      </section>
      <section class="panel">
        <h2>Station Utilization</h2>
        <div class="small" id="station-counts">Real OSM: 0 · Registered: 0 · Active: 0</div>
        <div id="util-bars" class="bar-row"></div>
      </section>
    </div>
  </div>
  <script>
    function fmt(value, fallback = 'N/A') {
      if (value === null || value === undefined || value === '') return fallback;
      if (typeof value === 'number') {
        if (!Number.isFinite(value)) return fallback;
        return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
      }
      return String(value);
    }
    function toNumber(value) {
      if (value === null || value === undefined || value === '') return null;
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    function normalizeCoordinate(value, min, max, padding, size) {
      const numeric = toNumber(value);
      if (numeric === null) return padding + (size / 2);
      const range = max - min;
      if (!Number.isFinite(range) || range <= 0) return padding + (size / 2);
      return padding + ((numeric - min) / range) * size;
    }
    let stationElementCache = new Map();
    let vehicleElementCache = new Map();
    let connectionPathCache = new Map();

    function createSvgElement(tag, attrs = {}) {
      const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
      Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
      return element;
    }

    function getStationPosition(station) {
      const stationX = toNumber(station.x);
      const stationY = toNumber(station.y);
      if (stationX !== null && stationY !== null) {
        return { x: stationX, y: stationY };
      }
      const lon = toNumber(station.lon);
      const lat = toNumber(station.lat);
      if (lon !== null && lat !== null) {
        return { x: lon, y: lat };
      }
      return null;
    }

    function getVehiclePosition(vehicle) {
      const vehicleX = toNumber(vehicle.x);
      const vehicleY = toNumber(vehicle.y);
      if (vehicleX !== null && vehicleY !== null) {
        return { x: vehicleX, y: vehicleY };
      }
      const currentPosition = vehicle.current_position || {};
      const currentX = toNumber(currentPosition.x ?? currentPosition.lon);
      const currentY = toNumber(currentPosition.y ?? currentPosition.lat);
      if (currentX !== null && currentY !== null) {
        return { x: currentX, y: currentY };
      }
      return null;
    }

    function renderNetwork(state) {
      const svg = document.getElementById('network-svg');
      const width = 680;
      const height = 360;
      const padding = 40;

      svg.innerHTML = '';
      const background = createSvgElement('rect', { x: '0', y: '0', width: String(width), height: String(height), fill: '#0e1725', rx: '16' });
      const layerGroup = createSvgElement('g', { id: 'network-layer' });
      const connectionLayer = createSvgElement('g', { id: 'connection-layer' });
      const stationLayer = createSvgElement('g', { id: 'station-layer' });
      const vehicleLayer = createSvgElement('g', { id: 'vehicle-layer' });
      layerGroup.appendChild(connectionLayer);
      layerGroup.appendChild(stationLayer);
      layerGroup.appendChild(vehicleLayer);
      svg.appendChild(background);
      svg.appendChild(layerGroup);

      const stations = Array.isArray(state.station_details) ? state.station_details : (Array.isArray(state.stations) ? state.stations : []);
      const vehicles = Array.isArray(state.vehicles) ? state.vehicles : [];
      const tracked = state.tracked_ev || {};
      const trackedVehicleId = tracked.id || tracked.vehicle_id || null;
      const selectedStationId = (state.ppo_decision && (state.ppo_decision.selected_station || state.ppo_decision.station)) || (state.ppo && (state.ppo.selected_station || state.ppo.station_id)) || (state.station && state.station.id) || null;

      const stationNodes = stations.map((station) => {
        const position = getStationPosition(station);
        return {
          station,
          position,
          id: String(station.id || station.station_id || 'station'),
        };
      }).filter((node) => node.position !== null);

      const vehicleNodes = vehicles.map((vehicle) => {
        const position = getVehiclePosition(vehicle);
        return {
          vehicle,
          position,
          id: String(vehicle.id || vehicle.vehicle_id || 'vehicle'),
        };
      }).filter((node) => node.position !== null);

      const allX = stationNodes.map((node) => node.position.x).concat(vehicleNodes.map((node) => node.position.x));
      const allY = stationNodes.map((node) => node.position.y).concat(vehicleNodes.map((node) => node.position.y));
      const xMin = Math.min(...allX, 0);
      const xMax = Math.max(...allX, 1);
      const yMin = Math.min(...allY, 0);
      const yMax = Math.max(...allY, 1);

      const normalizedStations = stationNodes.map((node) => ({
        ...node,
        x: normalizeCoordinate(node.position.x, xMin, xMax, padding, width - (padding * 2)),
        y: normalizeCoordinate(node.position.y, yMin, yMax, padding, height - (padding * 2)),
      }));
      const normalizedVehicles = vehicleNodes.map((node) => ({
        ...node,
        x: normalizeCoordinate(node.position.x, xMin, xMax, padding, width - (padding * 2)),
        y: normalizeCoordinate(node.position.y, yMin, yMax, padding, height - (padding * 2)),
      }));

      const stationLookup = new Map(normalizedStations.map((node) => [String(node.station.id || node.station.station_id || node.id), node]));
      const activeStationIds = new Set(stationLookup.keys());
      const activeVehicleIds = new Set(normalizedVehicles.map((node) => node.id));
      const activeConnectionKeys = new Set();

      for (const [key, element] of Array.from(stationElementCache.entries())) {
        if (!activeStationIds.has(key)) {
          element.remove();
          stationElementCache.delete(key);
        }
      }
      for (const [key, element] of Array.from(vehicleElementCache.entries())) {
        if (!activeVehicleIds.has(key)) {
          element.remove();
          vehicleElementCache.delete(key);
        }
      }
      for (const [key, element] of Array.from(connectionPathCache.entries())) {
        if (!activeConnectionKeys.has(key)) {
          element.remove();
          connectionPathCache.delete(key);
        }
      }

      normalizedStations.forEach((node) => {
        const stationId = String(node.station.id || node.station.station_id || node.id);
        let group = stationElementCache.get(stationId);
        if (!group) {
          group = createSvgElement('g', { 'data-station-id': stationId });
          stationLayer.appendChild(group);
          stationElementCache.set(stationId, group);
        }
        group.innerHTML = '';
        const isSelected = stationId === String(selectedStationId || '');
        const circle = createSvgElement('circle', {
          cx: String(node.x),
          cy: String(node.y),
          r: isSelected ? '14' : '10',
          fill: isSelected ? '#ffb84d' : '#6ea8ff',
          stroke: '#e5f0ff',
          'stroke-width': '2',
        });
        group.appendChild(circle);
        const label = createSvgElement('text', {
          x: String(node.x + 14),
          y: String(node.y - 12),
          fill: '#e5f0ff',
          'font-size': '12',
        });
        label.textContent = stationId;
        group.appendChild(label);
      });

      normalizedVehicles.forEach((node) => {
        const vehicleId = node.id;
        let group = vehicleElementCache.get(vehicleId);
        if (!group) {
          group = createSvgElement('g', { 'data-vehicle-id': vehicleId });
          vehicleLayer.appendChild(group);
          vehicleElementCache.set(vehicleId, group);
        }
        group.innerHTML = '';
        const isTracked = vehicleId === String(trackedVehicleId || '');
        const isSelected = Boolean(node.vehicle.recommended_station && selectedStationId && String(node.vehicle.recommended_station) === String(selectedStationId));
        const marker = createSvgElement('circle', {
          cx: String(node.x),
          cy: String(node.y),
          r: isTracked ? '8' : '6',
          fill: isTracked ? '#3dd9c7' : '#7fd3ff',
          stroke: isTracked ? '#d8fffb' : '#e8f7ff',
          'stroke-width': isTracked ? '2.5' : '1.5',
          opacity: '0.95',
        });
        group.appendChild(marker);
        if (isTracked) {
          const halo = createSvgElement('circle', {
            cx: String(node.x),
            cy: String(node.y),
            r: '12',
            fill: 'none',
            stroke: '#3dd9c7',
            'stroke-width': '1.5',
            'stroke-dasharray': '4 3',
            opacity: '0.85',
          });
          group.appendChild(halo);
        }
        const label = createSvgElement('text', {
          x: String(node.x + 10),
          y: String(node.y - 10),
          fill: '#8ea8c3',
          'font-size': '11',
        });
        label.textContent = vehicleId;
        group.appendChild(label);
        if (selectedStationId && isTracked) {
          const targetNode = stationLookup.get(String(selectedStationId));
          if (targetNode) {
            const connectionKey = `${vehicleId}:${selectedStationId}`;
            let path = connectionPathCache.get(connectionKey);
            if (!path) {
              path = createSvgElement('line', {
                'data-connection-key': connectionKey,
                stroke: '#ffb84d',
                'stroke-width': '2',
                'stroke-dasharray': '5 4',
                opacity: '0.9',
              });
              connectionLayer.appendChild(path);
              connectionPathCache.set(connectionKey, path);
            }
            path.setAttribute('x1', String(node.x));
            path.setAttribute('y1', String(node.y));
            path.setAttribute('x2', String(targetNode.x));
            path.setAttribute('y2', String(targetNode.y));
            activeConnectionKeys.add(connectionKey);
          }
        }
      });

      for (const [key, element] of Array.from(connectionPathCache.entries())) {
        if (!activeConnectionKeys.has(key)) {
          element.remove();
          connectionPathCache.delete(key);
        }
      }
    }
    function renderRewardChart(state) {
      const svg = document.getElementById('reward-chart');
      const rewardSeries = Array.isArray(state.charts && state.charts.reward_series) ? state.charts.reward_series : [];
      const rewards = rewardSeries.map((value) => toNumber(value.reward)).filter((value) => value !== null);
      svg.innerHTML = '';
      const width = 340;
      const height = 180;
      const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      bg.setAttribute('x', '0');
      bg.setAttribute('y', '0');
      bg.setAttribute('width', width);
      bg.setAttribute('height', height);
      bg.setAttribute('fill', '#0e1725');
      bg.setAttribute('rx', '12');
      svg.appendChild(bg);
      if (!rewards.length) {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', width / 2);
        text.setAttribute('y', height / 2);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', '#8ea8c3');
        text.textContent = 'Waiting for PPO decisions';
        svg.appendChild(text);
        return;
      }
      const maxVal = Math.max(...rewards.map(v => Math.abs(v))) || 1;
      const points = rewards.map((value, index) => {
        const x = 24 + (index / Math.max(1, rewards.length - 1)) * (width - 48);
        const y = 150 - (value / maxVal) * 110;
        return `${x},${y}`;
      }).join(' ');
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      path.setAttribute('points', points);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', '#3dd9c7');
      path.setAttribute('stroke-width', '3');
      svg.appendChild(path);
    }
    function renderUtilBars(state) {
      const container = document.getElementById('util-bars');
      const stations = Array.isArray(state.charts && state.charts.station_series) ? state.charts.station_series : [];
      container.innerHTML = '';
      if (!stations.length) {
        container.innerHTML = '<div class="small">No station telemetry yet</div>';
        return;
      }
      stations.slice(0, 8).forEach((station) => {
        const totalPorts = toNumber(station.total_ports) || 0;
        const availablePorts = toNumber(station.available_ports) || 0;
        const occupied = Math.max(0, totalPorts - availablePorts);
        const utilization = totalPorts ? occupied / totalPorts : toNumber(station.utilization) || 0;
        const bar = document.createElement('div');
        bar.style.flex = '1';
        bar.style.display = 'flex';
        bar.style.flexDirection = 'column';
        bar.style.alignItems = 'center';
        bar.innerHTML = `<div class="bar" style="height:${Math.max(10, utilization * 100)}px; width:100%;"></div><span>${fmt(station.id || station.name || 'station')}</span>`;
        container.appendChild(bar);
      });
    }
    function normalizeSimulationStatus(simulation) {
      const raw = String(simulation && simulation.simulation_status ? simulation.simulation_status : simulation && simulation.status ? simulation.status : '').toUpperCase();
      if (raw === 'PLAY' || raw === 'RUNNING' || raw === 'LIVE') return 'RUNNING';
      if (raw === 'PAUSED' || raw === 'PAUSE' || raw === 'STOPPED') return 'PAUSED';
    if (raw === 'COMPLETED' || raw === 'FINISHED' || raw === 'DONE') return 'COMPLETED';
      if (raw === 'ERROR' || raw === 'FAILED') return 'ERROR';
      return simulation && simulation.running === false ? 'PAUSED' : 'RUNNING';
    }
    function normalizeConnectionStatus(simulation) {
      const raw = String(simulation && simulation.connection_status ? simulation.connection_status : '').toUpperCase();
      return raw === 'DISCONNECTED' ? 'DISCONNECTED' : 'LIVE';
    }
    function updateControlButtons(simulation) {
      const speedSelect = document.getElementById('speed-select');
      const controls = ['play', 'pause', 'reset'];
      const normalizedStatus = String(simulation && (simulation.status || simulation.simulation_status || 'play') || 'play').toLowerCase();
      controls.forEach((action) => {
        const button = document.querySelector(`button[data-action="${action}"]`);
        if (!button) return;
        if (action === 'play') {
          button.textContent = normalizedStatus === 'play' || normalizedStatus === 'running' ? '▶ Running' : '▶ Play';
        } else if (action === 'pause') {
          button.textContent = normalizedStatus === 'paused' ? '⏸ Paused' : '⏸ Pause';
        } else {
          button.textContent = '↻ Reset';
        }
      });
      if (speedSelect && simulation && simulation.speed !== undefined) {
        speedSelect.value = String(simulation.speed);
      }
    }
    async function sendControl(action, payload = {}) {
      try {
        const response = await fetch('/control', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action, ...payload }),
        });
        if (!response.ok) {
          throw new Error(`Control request failed (${response.status})`);
        }
        return await response.json();
      } catch (error) {
        console.error(error);
        return null;
      }
    }
    function updateUI(state) {
      const simulation = state.simulation || {};
      const traffic = state.traffic || {};
      const kpis = state.kpis || {};
      const stations = Array.isArray(state.stations_summary) ? state.stations_summary : (Array.isArray(state.stations) ? state.stations : []);
      const selectedStation = stations.find((station) => (station.id || station.station_id) === (state.station && state.station.id ? state.station.id : null)) || stations[0] || {};
      const tracked = state.tracked_ev || {};
      const ppoDecision = state.ppo_decision || state.ppo || {};
      document.getElementById('sim-speed').textContent = fmt(simulation.speed || kpis.avg_speed || traffic.average_speed || 'N/A');
      document.getElementById('sim-time').textContent = fmt(simulation.sim_time || simulation.time || 'N/A');
      document.getElementById('sim-vehicles').textContent = fmt(simulation.fleet_size || simulation.ev_count || state.tracked_ids?.length || 'N/A');
      document.getElementById('sim-evs').textContent = fmt(simulation.ev_count || 'N/A');
      document.getElementById('sim-congestion').textContent = fmt(simulation.congestion || traffic.congestion_percent || 'N/A');
      document.getElementById('sim-step').textContent = fmt(simulation.step || 'N/A');
      document.getElementById('sim-connection').textContent = fmt(normalizeConnectionStatus(simulation));
      document.getElementById('sim-status').textContent = fmt(normalizeSimulationStatus(simulation));
      document.getElementById('stat-recs').textContent = fmt(kpis.recommendations ?? 0);
      document.getElementById('stat-charging-started').textContent = fmt(kpis.charging_started ?? 0);
      document.getElementById('stat-charging-done').textContent = fmt(kpis.charging_completed ?? 0);
      document.getElementById('stat-queue-events').textContent = fmt(kpis.queue_events ?? 0);
      updateControlButtons(simulation);
      document.getElementById('kpi-speed').textContent = fmt(kpis.avg_speed || traffic.average_speed || 'N/A');
      document.getElementById('kpi-traffic').textContent = fmt(traffic.traffic_density || traffic.congestion_percent || 'N/A');
      document.getElementById('kpi-wait').textContent = fmt(kpis.avg_wait_min || 'N/A');
      document.getElementById('kpi-energy').textContent = fmt(kpis.energy_consumed_kwh || state.network && state.network.total_energy || 'N/A');
      document.getElementById('station-name').textContent = fmt((state.station && state.station.name) || selectedStation.name || selectedStation.station_id || 'N/A');
      document.getElementById('station-id').textContent = fmt((state.station && state.station.id) || selectedStation.station_id || selectedStation.id || 'N/A');
      document.getElementById('station-ports').textContent = `${fmt((state.station && state.station.ports_available) || selectedStation.ports_available || selectedStation.available_ports || 0)}/${fmt((state.station && state.station.ports_total) || selectedStation.ports_total || selectedStation.total_ports || 0)}`;
      document.getElementById('station-queue').textContent = fmt((state.station && state.station.queue) || selectedStation.queue || 'N/A');
      document.getElementById('station-price').textContent = fmt((state.station && state.station.price) || selectedStation.price || selectedStation.price_per_kwh || 'N/A');
      document.getElementById('station-wait').textContent = fmt((state.station && state.station.wait_minutes) || selectedStation.wait_minutes || selectedStation.waiting_time_min || 'N/A');
      document.getElementById('station-load').textContent = fmt((state.station && state.station.load_kw) || selectedStation.load_kw || selectedStation.grid_load_kw || 'N/A');
    document.getElementById('station-util').textContent = fmt((state.station && state.station.utilization) || selectedStation.utilization || 'N/A');
    const stationStatistics = state.station_statistics || {};
    document.getElementById('station-counts').textContent = `Real OSM: ${stationStatistics.real_osm ?? 0} · Registered: ${stationStatistics.registered ?? 0} · Active: ${stationStatistics.active ?? 0}`;
      document.getElementById('ppo-vehicle').textContent = fmt(ppoDecision.ev_id || ppoDecision.vehicle_id || 'N/A');
      document.getElementById('ppo-action').textContent = fmt(ppoDecision.action || 'N/A');
      document.getElementById('ppo-station').textContent = fmt(ppoDecision.selected_station || ppoDecision.station_id || 'N/A');
      document.getElementById('ppo-reward').textContent = fmt(ppoDecision.reward || 'N/A');
      document.getElementById('ppo-battery').textContent = fmt(ppoDecision.battery || 'N/A');
      document.getElementById('ppo-reason').textContent = fmt(ppoDecision.reason || 'N/A');
      document.getElementById('sim-speed').textContent = fmt(simulation.speed || kpis.avg_speed || traffic.average_speed || 'N/A');
      document.getElementById('sim-time').textContent = fmt(simulation.sim_time || simulation.time || 'N/A');
      const trackedId = tracked.id || tracked.vehicle_id || 'N/A';
      if (trackedId !== 'N/A') {
        document.getElementById('station-name').textContent = fmt((state.station && state.station.name) || selectedStation.name || selectedStation.station_id || 'N/A');
      }
      renderNetwork(state);
      renderRewardChart(state);
      renderUtilBars(state);
    }
    let lastSuccessfulRefresh = 0;
    async function refresh() {
      try {
        const response = await fetch('dashboard_state.json?_=' + Date.now());
        if (!response.ok) {
          document.getElementById('sim-connection').textContent = 'DISCONNECTED';
          return;
        }
        const state = await response.json();
        lastSuccessfulRefresh = Date.now();
        updateUI(state);
            if (normalizeConnectionStatus(state.simulation) !== 'DISCONNECTED') {
                document.getElementById('sim-connection').textContent = 'LIVE';
            }
      } catch (error) {
        document.getElementById('sim-connection').textContent = 'DISCONNECTED';
        console.error(error);
      }
    }
    document.querySelectorAll('button[data-action]').forEach((button) => {
      button.addEventListener('click', async () => {
        const action = button.getAttribute('data-action');
        if (action === 'play') {
          await sendControl('play');
        } else if (action === 'pause') {
          await sendControl('pause');
        } else if (action === 'reset') {
          await sendControl('reset');
        }
        await refresh();
      });
    });
    const speedSelect = document.getElementById('speed-select');
    if (speedSelect) {
      speedSelect.addEventListener('change', async (event) => {
        const value = Number(event.target.value);
        if (!Number.isFinite(value)) {
          return;
        }
        await sendControl('set_speed', { speed: value });
        await refresh();
      });
    }
    setInterval(() => {
      if (lastSuccessfulRefresh && Date.now() - lastSuccessfulRefresh > 3000) {
        document.getElementById('sim-connection').textContent = 'DISCONNECTED';
      }
    }, 1000);
    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""
        template = template.replace("__SIM_SPEED__", self._format_dashboard_value(simulation.get("speed") or kpis.get("avg_speed")))
        template = template.replace("__SIM_TIME__", self._format_dashboard_value(simulation.get("time")))
        template = template.replace("__SIM_VEHICLES__", self._format_dashboard_value(simulation.get("fleet_size") or simulation.get("ev_count")))
        template = template.replace("__SIM_EVS__", self._format_dashboard_value(simulation.get("ev_count")))
        template = template.replace("__SIM_CONGESTION__", self._format_dashboard_value(traffic.get("congestion_percent")))
        template = template.replace("__SIM_STEP__", self._format_dashboard_value(simulation.get("step")))
        template = template.replace("__SIM_CONNECTION__", self._format_dashboard_value(simulation.get("connection_status")))
        template = template.replace("__SIM_STATUS__", self._format_dashboard_value(simulation.get("simulation_status")))
        template = template.replace("__KPI_SPEED__", self._format_dashboard_value(kpis.get("avg_speed") or traffic.get("average_speed")))
        template = template.replace("__KPI_TRAFFIC__", self._format_dashboard_value(traffic.get("traffic_density") or traffic.get("congestion_percent")))
        template = template.replace("__KPI_WAIT__", self._format_dashboard_value(kpis.get("avg_wait_min")))
        template = template.replace("__KPI_ENERGY__", self._format_dashboard_value(kpis.get("energy_consumed_kwh")))
        template = template.replace("__STATION_NAME__", self._format_dashboard_value(selected_station.get("name") if selected_station else selected_station_id))
        template = template.replace("__STATION_ID__", self._format_dashboard_value(selected_station.get("station_id") if selected_station else selected_station_id))
        template = template.replace("__STATION_PORTS__", self._format_dashboard_value((selected_station.get("available_ports") if selected_station else None), "0") + "/" + self._format_dashboard_value((selected_station.get("total_ports") if selected_station else None), "0"))
        template = template.replace("__STATION_QUEUE__", self._format_dashboard_value(selected_station.get("queue") if selected_station else None))
        template = template.replace("__STATION_PRICE__", self._format_dashboard_value(selected_station.get("price_per_kwh") if selected_station else None))
        template = template.replace("__STATION_WAIT__", self._format_dashboard_value(selected_station.get("waiting_time_min") if selected_station else None))
        template = template.replace("__STATION_LOAD__", self._format_dashboard_value(selected_station.get("grid_load_kw") if selected_station else None))
        template = template.replace("__STATION_UTIL__", self._format_dashboard_value(selected_station.get("utilization") if selected_station else None))
        template = template.replace("__PPO_VEHICLE__", self._format_dashboard_value(latest_ppo.get("vehicle_id") if latest_ppo else None))
        template = template.replace("__PPO_ACTION__", self._format_dashboard_value(latest_ppo.get("action") if latest_ppo else None))
        template = template.replace("__PPO_STATION__", self._format_dashboard_value(latest_ppo.get("station_id") if latest_ppo else None))
        template = template.replace("__PPO_REWARD__", self._format_dashboard_value(latest_ppo.get("reward") if latest_ppo else None))
        template = template.replace("__PPO_BATTERY__", self._format_dashboard_value(latest_ppo.get("battery") if latest_ppo else None))
        template = template.replace("__PPO_REASON__", self._format_dashboard_value((latest_ppo.get("reason") or latest_ppo.get("decision_features") or latest_ppo.get("recommendation_reason")) if latest_ppo else None))
        return template

    def _write_html(self) -> None:
        state = self._last_state or self._sample_state()
        html = self._build_digital_twin_html(state)
        self._atomic_write_text(self.html_path, html)

    def _write_research_dashboard(self, state: Dict[str, Any]) -> None:
        raw_summary = state.get("summary")
        summary = cast(dict[str, Any], raw_summary) if isinstance(raw_summary, dict) else {}
        raw_metrics = state.get("metrics")
        metrics_raw_list = cast(list[Any], raw_metrics) if isinstance(raw_metrics, list) else []
        metrics: list[dict[str, Any]] = [item for item in metrics_raw_list if isinstance(item, dict)]
        latest: dict[str, Any] = metrics[-1] if metrics else {}
        html = self._build_digital_twin_html(state)
        self._atomic_write_text(self.research_dashboard_path, html)
