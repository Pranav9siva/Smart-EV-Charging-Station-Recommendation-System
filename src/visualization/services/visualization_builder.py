from __future__ import annotations

import logging
from typing import Any

from src.visualization.map.renderer import battery_color, station_status_color, station_status_from_metrics
from src.visualization.services.road_network_service import RoadNetworkService
from src.xai.explainer import RecommendationExplainer

logger = logging.getLogger(__name__)


class VisualizationSnapshotBuilder:
    """Builds a visualization-friendly snapshot from the controller state."""

    def __init__(self, controller: Any) -> None:
        self.controller = controller
        self.explainer = RecommendationExplainer()
        self.road_service = RoadNetworkService()

    def build(self) -> dict[str, Any]:
        vehicles = []
        stations = []
        station_lookup = {}
        waiting_by_station: dict[str, int] = {}
        assigned_by_vehicle: dict[str, Any] = getattr(self.controller, "assignments", {}) or {}
        for vehicle_id, assignment in assigned_by_vehicle.items():
            if getattr(assignment, "status", "") == "waiting" and getattr(assignment, "station_id", None):
                sid = str(assignment.station_id)
                waiting_by_station[sid] = waiting_by_station.get(sid, 0) + 1
        network = self.road_service.build_payload()
        charging_station_ids: set[str] = set()
        try:
            import traci

            charging_station_ids = set(traci.chargingstation.getIDList())
        except Exception:
            pass
        snapshots = []
        snapshot_provider = getattr(self.controller, "_get_station_snapshots", None)
        if callable(snapshot_provider):
            snapshots = list(snapshot_provider(force_refresh=False) or [])
        if not snapshots:
            snapshots = list(self.controller.station_manager.get_station_snapshots())

        for station_state in snapshots:
            sid = str(station_state.get("station_id"))
            metrics = station_state
            waiting_count = waiting_by_station.get(sid, 0)
            station_payload = {
                "station_id": sid,
                "lat": station_state.get("lat"),
                "lon": station_state.get("lon"),
                "available_ports": metrics.get("available_ports"),
                "occupied_ports": metrics.get("occupied_ports"),
                "total_ports": metrics.get("total_ports"),
                "price_per_kwh": metrics.get("price_per_kwh"),
                "avg_wait_min": metrics.get("avg_wait_estimate"),
                "grid_load_kw": metrics.get("power_kw", 0.0),
                "status": station_status_from_metrics(metrics),
                "marker_color": station_status_color(station_status_from_metrics(metrics)),
                "queue_length": waiting_count,
                "charging_state": "active" if sid in charging_station_ids or int(metrics.get("occupied_ports", 0) or 0) > 0 or waiting_count > 0 else "idle",
            }
            if station_payload["lat"] is not None and station_payload["lon"] is not None:
                try:
                    controller_net = getattr(self.controller, "net", None)
                    if controller_net is None:
                        raise RuntimeError("SUMO network is not loaded")
                    x, y = controller_net.convertLonLat2XY(
                        float(station_payload["lon"]),
                        float(station_payload["lat"]),
                    )
                    station_payload["position"] = {"x": float(x), "z": float(y)}
                except Exception:
                    try:
                        x, y = self.controller.net.convertLonLat2XY(float(station_payload["lon"]), float(station_payload["lat"]))
                        station_payload["position"] = {"x": float(x), "z": float(y)}
                    except Exception:
                        geo_bounds = network.get("geo_bounds", {})
                        try:
                            lon_span = float(geo_bounds["maxLon"]) - float(geo_bounds["minLon"])
                            lat_span = float(geo_bounds["maxLat"]) - float(geo_bounds["minLat"])
                            x_span = float(geo_bounds["maxX"]) - float(geo_bounds["minX"])
                            y_span = float(geo_bounds["maxY"]) - float(geo_bounds["minY"])
                            x = float(geo_bounds["minX"]) + ((float(station_payload["lon"]) - float(geo_bounds["minLon"])) / lon_span) * x_span
                            y = float(geo_bounds["minY"]) + ((float(station_payload["lat"]) - float(geo_bounds["minLat"])) / lat_span) * y_span
                            station_payload["position"] = {"x": x, "z": y}
                        except (KeyError, TypeError, ValueError, ZeroDivisionError):
                            logger.warning("[STATIONS] failed coordinate conversion station=%s", sid)
            stations.append(station_payload)
            station_lookup[sid] = station_payload

        active_sumo_ids: set[str] = set()
        try:
            import traci

            active_sumo_ids = set(traci.vehicle.getIDList())
        except Exception:
            pass

        visualized_sumo_ids: set[str] = set()
        for vehicle in self.controller.vehicle_manager.list_vehicles():
            vdict = vehicle.to_dict()
            assignment = assigned_by_vehicle.get(vehicle.vehicle_id)
            charging_status = getattr(assignment, "status", None) if assignment else None
            vdict.update(
                {
                    "id": vehicle.vehicle_id,
                    "battery_pct": round(float(vehicle.battery_pct), 2),
                    "marker_color": battery_color(float(vehicle.battery_pct)),
                    "charging_status": charging_status or "driving",
                }
            )
            sumo_vid = getattr(self.controller, "vm_to_sumo", {}).get(vehicle.vehicle_id, vehicle.vehicle_id)
            if sumo_vid in active_sumo_ids:
                try:
                    import traci

                    pos = traci.vehicle.getPosition(sumo_vid)
                    if pos and len(pos) >= 2:
                        vdict["current_position"] = {"lat": float(pos[1]), "lon": float(pos[0])}
                        vdict["speed"] = float(traci.vehicle.getSpeed(sumo_vid))
                        vdict["heading_deg"] = float(traci.vehicle.getAngle(sumo_vid))
                        vdict["current_edge"] = traci.vehicle.getRoadID(sumo_vid)
                        vdict["vehicle_type"] = traci.vehicle.getTypeID(sumo_vid)
                        vdict["is_ev"] = True
                        vdict["tracked"] = bool(getattr(vehicle, "tracked", False))
                        visualized_sumo_ids.add(sumo_vid)
                except Exception:
                    pass

            position = vdict.get("current_position")
            if not isinstance(position, dict) or position.get("lat") is None or position.get("lon") is None:
                continue

            destination_station = getattr(assignment, "station_id", None) if assignment else None
            station = station_lookup.get(destination_station) if destination_station else None
            if station is not None:
                vdict["destination_path"] = [
                    {"lon": vdict.get("current_position", {}).get("lon", 0.0), "lat": vdict.get("current_position", {}).get("lat", 0.0)},
                    {"lon": station.get("lon", 0.0), "lat": station.get("lat", 0.0)},
                ]
                vdict["charging_target"] = station.get("station_id")
            else:
                vdict["destination_path"] = []
                vdict["charging_target"] = None
            vehicles.append(vdict)

        for sumo_vid in sorted(active_sumo_ids - visualized_sumo_ids):
            try:
                import traci

                pos = traci.vehicle.getPosition(sumo_vid)
                if not pos or len(pos) < 2:
                    logger.warning("[VEHICLES] malformed position for SUMO vehicle=%s", sumo_vid)
                    continue
                vehicles.append({
                    "id": sumo_vid,
                    "vehicle_id": sumo_vid,
                    "is_ev": False,
                    "tracked": False,
                    "vehicle_type": traci.vehicle.getTypeID(sumo_vid),
                    "current_position": {"lat": float(pos[1]), "lon": float(pos[0])},
                    "speed": float(traci.vehicle.getSpeed(sumo_vid)),
                    "heading_deg": float(traci.vehicle.getAngle(sumo_vid)),
                    "current_edge": traci.vehicle.getRoadID(sumo_vid),
                    "battery_pct": None,
                    "charging_status": "traffic",
                    "destination_path": [],
                    "charging_target": None,
                })
            except Exception:
                logger.exception("[VEHICLES] failed to serialize SUMO vehicle=%s", sumo_vid)

        logger.info("[VEHICLES] active_sumo=%d visualized=%d", len(active_sumo_ids), len(vehicles))

        summary = {
            "charging_vehicles": sum(1 for vehicle in vehicles if str(vehicle.get("charging_status") or "").lower() == "charging"),
            "active_assignments": len(self.controller.assignments),
            "total_vehicles": len(vehicles),
            "simulation_time": getattr(self.controller, "current_step", None),
        }

        station_available = sum(int(station.get("available_ports", 0) or 0) for station in stations)
        station_occupied = sum(int(station.get("occupied_ports", 0) or 0) for station in stations)
        station_queue_total = sum(int(station.get("queue_length", 0) or 0) for station in stations)

        avg_speed = 0.0
        if vehicles:
            speed_values = [float(vehicle.get("speed", 0.0) or 0.0) for vehicle in vehicles if vehicle.get("speed") is not None]
            if speed_values:
                avg_speed = float(sum(speed_values) / max(1, len(speed_values)))
        slow_vehicles = sum(1 for vehicle in vehicles if float(vehicle.get("speed", 0.0) or 0.0) <= 2.0)
        congestion_percent = (slow_vehicles / max(1, len(vehicles))) * 100.0 if vehicles else 0.0

        edge_groups: dict[str, list[float]] = {}
        for vehicle in vehicles:
            edge_id = vehicle.get("current_edge")
            if not edge_id:
                continue
            edge_groups.setdefault(str(edge_id), []).append(float(vehicle.get("speed", 0.0) or 0.0))
        max_edge_count = max((len(values) for values in edge_groups.values()), default=1)
        traffic = [
            {
                "edge_id": edge_id,
                "congestion": round(len(values) / max(1, max_edge_count), 4),
                "speed": round(float(sum(values) / max(1, len(values))), 4),
            }
            for edge_id, values in edge_groups.items()
        ]
        traffic.sort(key=lambda item: item["congestion"], reverse=True)

        tracked_ids = [vehicle.vehicle_id for vehicle in self.controller.vehicle_manager.list_tracked_vehicles()]

        latest_metric = self.controller.simulation_metrics[-1] if self.controller.simulation_metrics else {}
        avg_wait_min = float(latest_metric.get("avg_wait_min", 0.0) or 0.0)

        energy_consumed_kwh = 0.0
        initial_battery = getattr(self.controller, "_initial_battery_snapshot", {}) or {}
        for vehicle in self.controller.vehicle_manager.list_vehicles():
            initial = initial_battery.get(vehicle.vehicle_id)
            if initial is None:
                continue
            initial_pct, capacity_kwh = initial
            consumed = max(0.0, (float(initial_pct) - float(vehicle.battery_pct)) / 100.0 * float(capacity_kwh))
            energy_consumed_kwh += consumed

        latest_recommendation = None
        if self.controller.recommendation_log:
            latest_recommendation = self.controller.recommendation_log[-1]
            selected_station = latest_recommendation.get("selected_station")
            if selected_station and selected_station in station_lookup:
                station_lookup[selected_station]["is_recommended"] = True

        ppo = {
            "ev_id": latest_recommendation.get("vehicle_id") if latest_recommendation else None,
            "action": (latest_recommendation.get("ppo_action") if latest_recommendation.get("ppo_action") is not None else latest_recommendation.get("selected_station")) if latest_recommendation else None,
            "ppo_action": latest_recommendation.get("ppo_action") if latest_recommendation else None,
            "station_id": latest_recommendation.get("selected_station") if latest_recommendation else None,
            "reward": latest_recommendation.get("ppo_reward") if latest_recommendation else None,
            "battery": latest_recommendation.get("battery_pct") if latest_recommendation else None,
        }

        latest_explanation = None
        if getattr(self.controller, "last_explanation", None) is not None:
            latest_explanation = dict(self.controller.last_explanation)
        elif latest_recommendation:
            latest_explanation = None
        if latest_explanation is None and latest_recommendation:
            vehicle_state = {
                "battery_pct": float(latest_recommendation.get("battery_pct", 0.0)),
                "battery_capacity_kwh": float(latest_recommendation.get("battery_capacity_kwh", 0.0)),
                "remaining_range_km": 0.0,
            }
            recommendation_payload = {
                "station": latest_recommendation.get("selected_station"),
                "travel_distance": latest_recommendation.get("travel_distance", 0.0),
                "waiting_time": latest_recommendation.get("waiting_time", 0.0),
                "queue_length": latest_recommendation.get("queue_length", 0),
                "charging_cost": latest_recommendation.get("charging_cost", 0.0),
                "available_ports": latest_recommendation.get("available_ports", 0),
                "grid_load": latest_recommendation.get("grid_load", 0.0),
                "ppo_reward": latest_recommendation.get("ppo_reward", 0.0),
            }
            latest_explanation = self.explainer.explain_recommendation(
                vehicle_state=vehicle_state,
                recommendation=recommendation_payload,
                station_manager=self.controller.station_manager,
            )

        traffic_lights = self._build_traffic_lights(network, stations)
        logger.info("[STATIONS] visualized=%d", len(stations))

        control = {"status": "play", "speed": 1.0}
        runtime_control = getattr(self.controller, "runtime_control", None)
        if runtime_control is not None:
            try:
                control = runtime_control.read()
            except Exception:
                control = {"status": "play", "speed": 1.0}

        simulation = {
            "time": float(getattr(self.controller, "current_step", 0)),
            "step": int(getattr(self.controller, "current_step", 0)),
            "running": control.get("status") != "paused",
            "speed": float(control.get("speed", 1.0) or 1.0),
            "fleet_size": len(self.controller.vehicle_manager.list_vehicles()),
            "ev_count": len(self.controller.vehicle_manager.list_vehicles()),
            "tracked_count": len(tracked_ids),
            "congestion_percent": round(float(congestion_percent), 4),
        }

        kpis = {
            "avg_speed": round(float(avg_speed), 4),
            "traffic_density": round(float(len(vehicles) / max(1, len(edge_groups))) if edge_groups else 0.0, 4),
            "avg_wait_min": round(float(avg_wait_min), 4),
            "energy_consumed_kwh": round(float(energy_consumed_kwh), 4),
        }

        stations_aggregate = {
            "total": len(stations),
            "available": station_available,
            "occupied": station_occupied,
            "queue_total": station_queue_total,
        }
        station_statistics = {
            "real_osm": sum(
                1 for station in stations
                if str(station.get("station_id") or "").lower().startswith("osm_")
                or "osm" in str(station.get("data_source") or "").lower()
            ),
            "registered": len(stations),
            "active": sum(
                1 for station in stations
                if int(station.get("occupied_ports", 0) or 0) > 0
                or int(station.get("queue_length", 0) or 0) > 0
                or str(station.get("charging_state") or "").lower() in {"active", "charging", "queued"}
            ),
        }

        return {
            "simulation": simulation,
            "kpis": kpis,
            "stations": stations_aggregate,
            "station_statistics": station_statistics,
            "vehicles": vehicles,
            "station_details": stations,
            "traffic": traffic,
            "ppo": ppo,
            "network": network,
            "traffic_lights": traffic_lights,
            "summary": summary,
            "metrics": list(self.controller.simulation_metrics[-20:]),
            "tracked_ids": tracked_ids,
            "latest_recommendation": latest_recommendation,
            "recommendations": self.controller.recommendation_log[-20:],
            "charging_events": self.controller.charging_events[-50:],
            "xai": {
                "latest_explanation": latest_explanation or {},
                "explanation_count": len(self.controller.recommendation_log),
            },
        }

    def _build_traffic_lights(self, network: dict[str, Any], stations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        edges = network.get("edges", []) or []
        lights = []
        for index, edge in enumerate(edges[:6]):
            points = edge.get("points", []) or []
            if not points:
                continue
            first = points[0]
            lights.append({"id": f"tl_{index}", "x": first[0], "z": first[1], "state": "green"})
        if not lights and stations:
            for index, station in enumerate(stations[:3]):
                lights.append({"id": f"tl_{index}", "x": station.get("lon", 0.0), "z": station.get("lat", 0.0), "state": "green"})
        return lights
