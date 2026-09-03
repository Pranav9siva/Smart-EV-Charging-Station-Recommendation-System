from __future__ import annotations

import json
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from src.monitoring.metrics import increment_counter, update_metric

from .station_repository import StationRepository


class ChargingStationManager:
    """Runtime manager that exposes SUMO and SQLite-backed charging station operations."""

    def __init__(self, db_path: str | Path | None = None, connection: Connection | None = None) -> None:
        if connection is not None:
            self.repo = StationRepository(connection)
        else:
            if db_path is None:
                raise ValueError("db_path or connection required")
            self.repo = StationRepository(str(db_path))
        self._station_snapshot_cache: list[dict[str, Any]] = []
        self._station_snapshot_cache_by_id: dict[str, dict[str, Any]] = {}

    def _invalidate_station_snapshot_cache(self) -> None:
        self._station_snapshot_cache = []
        self._station_snapshot_cache_by_id = {}

    def list_all_stations(self) -> list[dict[str, Any]]:
        rows = self.repo.get_all_stations()
        return [
            {
                "station_id": row[0],
                "name": row[1],
                "lat": row[2],
                "lon": row[3],
                "operator": row[4],
                "address": row[5],
                "grid_zone_id": row[6],
                "data_source": row[7],
            }
            for row in rows
        ]

    def get_station_state(self, station_id: str) -> dict[str, Any] | None:
        return self.get_station_snapshot(station_id)

    def get_station_snapshots(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        if not force_refresh and self._station_snapshot_cache:
            return self._station_snapshot_cache
        snapshots = self.repo.get_station_snapshots()
        self._station_snapshot_cache = snapshots
        self._station_snapshot_cache_by_id = {str(item.get("station_id")): item for item in snapshots}
        return snapshots

    def get_station_snapshot(self, station_id: str, force_refresh: bool = False) -> dict[str, Any] | None:
        if not station_id:
            return None
        if force_refresh or not self._station_snapshot_cache_by_id:
            self.get_station_snapshots(force_refresh=force_refresh)
        return self._station_snapshot_cache_by_id.get(str(station_id))

    def get_nearby(self, lat: float, lon: float, radius_km: float = 2.0) -> list[dict[str, Any]]:
        return self.repo.get_nearby_stations(lat, lon, radius_km)

    def get_station_metrics(self, station_id: str) -> dict[str, Any] | None:
        station = self.get_station_snapshot(station_id)
        if station is None:
            return None
        return {
            "station_id": station_id,
            "available_ports": int(station.get("available_ports", 0) or 0),
            "occupied_ports": int(station.get("occupied_ports", 0) or 0),
            "total_ports": int(station.get("total_ports", 0) or 0),
            "charging_ports": int(station.get("charging_ports", 0) or 0),
            "queue_length": int(station.get("queue_length", 0) or 0),
            "avg_wait_estimate": float(station.get("avg_wait_estimate", 0.0) or 0.0),
            "price_per_kwh": station.get("price_per_kwh"),
            "ports": station.get("ports", []),
            "power_kw": float(station.get("grid_load_kw", 0.0) or 0.0),
            "occupancy_rate": float(station.get("occupancy_rate", 0.0) or 0.0),
            "occupancy_percent": float(station.get("occupancy_percent", 0.0) or 0.0),
            "charging_state": station.get("charging_state"),
        }

    def get_available_ports(self, station_id: str) -> list[str]:
        station = self.get_station_snapshot(station_id)
        if station is None:
            return []
        return [str(port["port_id"]) for port in station.get("ports", []) if port.get("status") == "Available"]

    def reserve_port(self, station_id: str, reserved_status: str = "Charging") -> str | None:
        available_ports = self.get_available_ports(station_id)
        if not available_ports:
            increment_counter("rejected_charging")
            return None
        port_id = available_ports[0]
        self.update_port_status(port_id, reserved_status)
        increment_counter("charging_requests")
        update_metric("charging_queue_length", len(self.get_station_snapshots()) + 1)
        return port_id

    def release_port(self, port_id: str) -> None:
        self.update_port_status(port_id, "Available")
        increment_counter("successful_charging")

    def reset_all_ports_to_available(self) -> dict[str, int]:
        """Reset transient persisted charging state for a fresh simulation startup."""
        summary = self.repo.reset_ports_for_new_simulation()
        self._invalidate_station_snapshot_cache()
        return summary

    def update_port_status(self, port_id: str, status: str) -> None:
        station_id = self.repo.get_station_id_for_port(port_id)
        self.repo.log_status_change(port_id, status)
        self._invalidate_station_snapshot_cache()
        try:
            state = self.get_station_snapshot(station_id, force_refresh=True) if station_id else None
            if state:
                available = int(state.get("available_ports", 0) or 0)
                occupied = int(state.get("occupied_ports", 0) or 0)
                total_ports = int(state.get("total_ports", 0) or 0)
                update_metric("available_ports", available)
                update_metric("occupied_ports", occupied)
                update_metric("station_utilization_percent", (occupied / max(1, total_ports)) * 100.0)
                update_metric("station_utilization", occupied / max(1, total_ports))
        except Exception:
            pass

    def generate_sumo_additional_file(
        self,
        net_path: str | Path,
        output_path: str | Path,
        lookup_path: str | Path | None = None,
        max_snap_distance_m: float = 100.0,
    ) -> list[dict[str, Any]]:
        import sumolib

        net = sumolib.net.readNet(str(net_path))
        stations: list[dict[str, Any]] = []
        for station in self.list_all_stations():
            try:
                lane_id, start_pos = self._snap_station_to_lane(net, station["lat"], station["lon"], max_snap_distance_m)
            except RuntimeError:
                continue
            stations.append(
                {
                    "station_id": station["station_id"],
                    "name": station["name"],
                    "lat": station["lat"],
                    "lon": station["lon"],
                    "operator": station["operator"],
                    "address": station["address"],
                    "data_source": station["data_source"],
                    "lane_id": lane_id,
                    "start_pos": round(start_pos, 1),
                    "power_kw": self._get_station_power(station["station_id"]),
                }
            )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        lines = ["<additional>"]
        for station in stations:
            lines.append(
                f'  <chargingStation id="{station["station_id"]}" lane="{station["lane_id"]}" startPos="{station["start_pos"]}" power="{station["power_kw"]}" chargeInTransit="false"/>'
            )
        lines.append("</additional>")
        output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if lookup_path is not None:
            lookup_file = Path(lookup_path)
            lookup_file.parent.mkdir(parents=True, exist_ok=True)
            lookup_file.write_text(json.dumps({station["station_id"]: station for station in stations}, indent=2), encoding="utf-8")

        return stations

    def _get_station_power(self, station_id: str) -> float:
        state = self.get_station_state(station_id)
        if not state:
            return 22.0
        ports = state.get("ports", [])
        if not ports:
            return 22.0
        power_values = [float(port.get("power_kw", 22.0)) for port in ports]
        return max(power_values) if power_values else 22.0

    def _snap_station_to_lane(self, net, lat: float, lon: float, max_snap_distance_m: float) -> tuple[str, float]:
        x, y = net.convertLonLat2XY(lon, lat)
        neighboring_lanes = net.getNeighboringLanes(x, y, max_snap_distance_m)
        if not neighboring_lanes:
            raise RuntimeError(f"Could not snap station at ({lat},{lon}) to the SUMO network")
        lane_ref, _ = min(neighboring_lanes, key=lambda item: item[1])
        # sumolib may return either a lane-id string or a Lane object depending on index backend.
        if hasattr(lane_ref, "getID") and hasattr(lane_ref, "getLength"):
            lane = lane_ref
            lane_id = lane.getID()
        else:
            lane_id = str(lane_ref)
            lane = net.getLane(lane_id)
        lane_len = lane.getLength()
        if lane_len <= 0.5:
            start_pos = max(0.01, lane_len / 2.0)
        else:
            start_pos = min(max(0.01, lane_len * 0.1), max(0.01, lane_len - 0.01))
        return lane_id, start_pos

    def _station_id_for_port(self, port_id: str) -> str:
        station_id = self.repo.get_station_id_for_port(port_id)
        if station_id:
            return station_id
        for station in self.get_station_snapshots():
            for port in station.get("ports", []):
                if port.get("port_id") == port_id:
                    return str(station["station_id"])
        return ""

    def close(self) -> None:
        self.repo.close()


# Maintain backward compatibility for imports that expect StationManager
StationManager = ChargingStationManager
