from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.core import Env

from sumolib import checkBinary

from src.recommendation_engine.engine import RecommendationEngine
from src.route_planning.network_graph import NetworkGraph
from src.station_management.station_repository import StationRepository
from src.station_management.station_state import ChargingStation, StationManager


class _StationManagerAdapter:
    """Compatibility helper for environments that need station-manager-like access."""

    def __init__(self, station_manager) -> None:
        self._station_manager = station_manager

    def get_station_state(self, station_id: str) -> Optional[dict]:
        if self._station_manager is None:
            return None
        return self._station_manager.get_station_state(station_id)

    def list_all_stations(self) -> list[dict]:
        if self._station_manager is None:
            return []
        return self._station_manager.list_all_stations()

    def update_port_status(self, port_id: str, status: str) -> None:
        if self._station_manager is not None:
            self._station_manager.update_port_status(port_id, status)


def _ensure_traci(sys_path_insert: Optional[Path] = None) -> None:
    # Ensure SUMO tools are on sys.path so traci can be imported.
    try:
        import traci  # type: ignore
        return
    except Exception:
        pass

    # Use sumolib to locate binary and infer SUMO_HOME
    bin_path = checkBinary("sumo")
    bin_path = Path(bin_path).resolve()
    sumo_home = bin_path.parents[1]
    os.environ.setdefault("SUMO_HOME", str(sumo_home))
    tools_path = sumo_home / "tools"
    if str(tools_path) not in sys.path:
        sys.path.insert(0, str(tools_path))


class TraciEVChargingEnv(Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        sumo_config: str | Path,
        tracked_vehicle_ids: list[str] | None = None,
        step_per_action: int = 10,
        db_path: Optional[str] = None,
        station_manager=None,
    ) -> None:
        _ensure_traci()
        import traci  # type: ignore

        self.traci = traci
        self.step_per_action = int(step_per_action)
        self.sumo_cfg = str(Path(sumo_config))

        # Prepare DB-backed station repo if available
        if db_path:
            self.repo = StationRepository(db_path)
        else:
            self.repo = None

        self.station_manager = station_manager
        if self.station_manager is None and db_path:
            from src.station_management.manager import StationManager

            self.station_manager = StationManager(db_path)

        self.station_adapter = _StationManagerAdapter(self.station_manager)

        # tracked vehicles list (detailed monitoring)
        self.tracked_vehicle_ids = tracked_vehicle_ids or []
        self._active_ports: dict[str, str] = {}
        self._step_index = 0
        self._recommendation_engine = None
        self._recommendation_history: list[dict] = []

        # observation: per-tracked vehicle: [lat, lon, speed, soc_est]
        vcount = max(1, len(self.tracked_vehicle_ids))
        self.observation_space = spaces.Dict(
            {
                "vehicles": spaces.Box(low=-1e6, high=1e6, shape=(vcount, 4), dtype=np.float32),
            }
        )
        # action: select station index from the list of nearby stations for the focused tracked vehicle
        self.action_space = spaces.Discrete(10)

        self._started = False

    def start(self, gui: bool = False) -> None:
        if self._started:
            return
        bin_path = checkBinary("sumo-gui" if gui else "sumo")
        cmd = [str(bin_path), "-c", self.sumo_cfg, "--no-step-log", "true"]
        self.traci.start(cmd)
        self._started = True

    def reset(self, *, seed=None, options=None):
        # start SUMO if not started
        if not self._started:
            self.start(gui=False)

        # build initial obs
        self._refresh_station_state()
        self._build_recommendation_engine()
        obs = self._build_observation()
        return obs, {}

    def _build_observation(self) -> dict:
        vehicles_obs = []
        for vid in self.tracked_vehicle_ids:
            try:
                x, y = self.traci.vehicle.getPosition(vid)
                speed = self.traci.vehicle.getSpeed(vid)
                # try battery parameter if available
                soc = 0.0
                try:
                    soc = float(self.traci.vehicle.getParameter(vid, "device.battery.actualChargePercent"))
                except Exception:
                    # fallback: unknown
                    soc = -1.0
                vehicles_obs.append([x, y, speed, soc])
            except Exception:
                vehicles_obs.append([0.0, 0.0, 0.0, -1.0])

        arr = np.array(vehicles_obs, dtype=np.float32)
        return {"vehicles": arr}

    def _build_recommendation_engine(self) -> None:
        if self._recommendation_engine is not None:
            return
        if self.station_manager is None:
            return
        try:
            network = NetworkGraph()
            for node in ["A", "B", "C", "D"]:
                network.add_node(node)
            network.add_edge("A", "B", travel_time=2.0, distance=1.0)
            network.add_edge("B", "C", travel_time=2.0, distance=1.0)
            network.add_edge("C", "D", travel_time=2.0, distance=1.0)
            network.add_edge("A", "D", travel_time=5.0, distance=2.0)
            stations = []
            for station in self.station_adapter.list_all_stations()[:5]:
                stations.append(ChargingStation(station["station_id"], capacity=2, occupancy=0, waiting_time=1.0))
            self._recommendation_engine = RecommendationEngine(StationManager(stations), network)
        except Exception:
            self._recommendation_engine = None

    def _refresh_station_state(self) -> None:
        if not self.station_adapter:
            return

        for station in self.station_adapter.list_all_stations():
            station_state = self.station_adapter.get_station_state(station["station_id"])
            if not station_state:
                continue

            available_ports = [p for p in station_state.get("ports", []) if p.get("status") == "Available"]
            if not available_ports:
                continue

            # assign one live charging slot per tracked vehicle when possible
            for vehicle_id in self.tracked_vehicle_ids:
                if not available_ports:
                    break
                assigned_port_ids = {port_id for port_id, assigned_vehicle in self._active_ports.items() if assigned_vehicle == vehicle_id}
                if assigned_port_ids:
                    continue
                port = available_ports.pop(0)
                self.station_adapter.update_port_status(port["port_id"], "Charging")
                self._active_ports[port["port_id"]] = vehicle_id

    def step(self, action):
        self._build_recommendation_engine()
        for _ in range(self.step_per_action):
            self.traci.simulationStep()

        self._refresh_station_state()
        self._step_index += 1

        obs = self._build_observation()
        reward = 0.0
        selected_station_id = None
        recommended_station_id = None
        if self._recommendation_engine is not None:
            try:
                candidate_stations = []
                for station in self.station_adapter.list_all_stations():
                    station_state = self.station_adapter.get_station_state(station["station_id"])
                    if not station_state:
                        continue
                    available_ports = sum(1 for port in station_state.get("ports", []) if port.get("status") == "Available")
                    if available_ports:
                        candidate_stations.append(StationCandidate(
                            station_id=station["station_id"],
                            distance_km=float(station_state.get("distance_km", 2.0)),
                            travel_time_min=float(station_state.get("travel_time_min", 5.0)),
                            free_ports=available_ports,
                            total_ports=max(1, len(station_state.get("ports", []))),
                            power_kw=float(station_state.get("price_per_kwh") or 22.0),
                        ))
                if candidate_stations:
                    recommended = self._recommendation_engine.recommend(
                        {"soc_pct": 50.0, "battery_kwh": 40.0, "consumption_wh_per_km": 200.0},
                        candidate_stations,
                        top_k=1,
                    )
                    recommended_station_id = recommended[0].station_id if recommended else None
                    selected_station_id = candidate_stations[min(action, len(candidate_stations) - 1)].station_id
                    self._recommendation_history.append({"step": self._step_index, "selected_station_id": selected_station_id, "recommended_station_id": recommended_station_id, "action": int(action)})
            except Exception:
                selected_station_id = None

        if self.station_adapter:
            for station in self.station_adapter.list_all_stations():
                station_state = self.station_adapter.get_station_state(station["station_id"])
                if not station_state:
                    continue
                available_ports = sum(1 for port in station_state.get("ports", []) if port.get("status") == "Available")
                reward += available_ports * 1.0
                if selected_station_id and selected_station_id == station["station_id"]:
                    reward += 3.0
                if recommended_station_id and station["station_id"] == recommended_station_id:
                    reward += 2.0
        if selected_station_id and recommended_station_id and selected_station_id == recommended_station_id:
            reward += 5.0
        if selected_station_id and recommended_station_id and selected_station_id != recommended_station_id:
            reward -= 1.5

        done = False
        info = {"step_index": self._step_index, "active_ports": self._active_ports, "selected_station_id": selected_station_id}
        return obs, float(reward), done, False, info

    def render(self):
        print("TraciEVChargingEnv: tracked vehicles:")
        obs = self._build_observation()
        print(obs)

    def close(self):
        try:
            if self._started:
                self.traci.close()
        except Exception:
            pass
        if self.station_manager is not None and hasattr(self.station_manager, "close"):
            try:
                self.station_manager.close()
            except Exception:
                pass
