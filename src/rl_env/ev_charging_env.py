from __future__ import annotations

import numpy as np
from gymnasium import spaces
from gymnasium.core import Env
from pathlib import Path
from sqlite3 import connect

from src.recommendation.engine import RecommendationEngine, StationCandidate
from src.station_management.station_repository import StationRepository


class EVChargingEnv(Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        db_path: Path | str | None = None,
        radius_km: float = 2.0,
        num_stations: int = 3,
        num_vehicles: int = 4,
        prefer_default_db: bool = True,
        max_episode_steps: int | None = None,
    ) -> None:
        self.radius_km = radius_km
        self.num_vehicles = num_vehicles

        if db_path is not None:
            self.db_path = Path(db_path)
            self.connection = connect(str(self.db_path))
        else:
            default_db = Path("data/stations/stations.sqlite")
            if prefer_default_db and default_db.exists():
                self.db_path = default_db
                self.connection = connect(str(self.db_path))
            else:
                self.db_path = None
                self.connection = connect(":memory:")
                self._seed_in_memory_db(num_stations)

        self.repo = StationRepository(self.connection)
        self.stations = self.repo.get_all_stations()
        self.station_ids = [station[0] for station in self.stations]
        self.num_stations = max(1, len(self.stations))
        self.recommendation_engine = RecommendationEngine()
        self.station_candidates = self._build_recommendation_candidates()
        if max_episode_steps is None:
            self.max_episode_steps = max(1, self.num_stations)
        else:
            self.max_episode_steps = max(1, int(max_episode_steps))

        self.observation_space = spaces.Dict(
            {
                "available_ports": spaces.Box(low=0, high=100, shape=(self.num_stations,), dtype=np.int32),
                "occupied_ports": spaces.Box(low=0, high=100, shape=(self.num_stations,), dtype=np.int32),
                "price_per_kwh": spaces.Box(low=0.0, high=100.0, shape=(self.num_stations,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(self.num_stations)
        self.state = self._build_observation()
        self.done = False
        self.last_selected_station: str | None = None
        self._step_index = 0

    def _seed_in_memory_db(self, num_stations: int) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS stations (
                station_id TEXT PRIMARY KEY,
                name TEXT,
                lat REAL,
                lon REAL,
                operator TEXT,
                address TEXT,
                grid_zone_id TEXT,
                data_source TEXT
            );
            CREATE TABLE IF NOT EXISTS ports (
                port_id TEXT PRIMARY KEY,
                station_id TEXT REFERENCES stations(station_id),
                connector_type TEXT,
                power_kw REAL,
                status TEXT,
                status_updated_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT REFERENCES stations(station_id),
                price_per_kwh REAL,
                tariff_period TEXT,
                recorded_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS occupancy_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                port_id TEXT REFERENCES ports(port_id),
                status TEXT,
                changed_at TIMESTAMP
            );
            """
        )
        for i in range(1, num_stations + 1):
            st_id = f"station_{i}"
            self.connection.execute(
                "INSERT INTO stations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (st_id, f"Station {i}", 12.9 + i * 0.01, 77.5 + i * 0.01, "Sim", "Address", f"zone_{i}", "simulated"),
            )
            for p in range(1, 3):
                port_id = f"{st_id}_p{p}"
                self.connection.execute(
                    "INSERT INTO ports VALUES (?, ?, ?, ?, ?, ?)",
                    (port_id, st_id, "Type2", 22.0, "Available", "2026-01-01T00:00:00Z"),
                )
            self.connection.execute(
                "INSERT INTO price_history (station_id, price_per_kwh, tariff_period, recorded_at) VALUES (?, ?, ?, ?)",
                (st_id, 15.0, "non_solar_hour", "2026-01-01T00:00:00Z"),
            )
        self.connection.commit()

    def _build_recommendation_candidates(self) -> list[StationCandidate]:
        candidates: list[StationCandidate] = []
        for station in self.stations:
            station_id = station[0]
            state = self.repo.get_station_state(station_id)
            if not state:
                continue
            available_ports = sum(1 for port in state["ports"] if port["status"] == "Available")
            total_ports = len(state["ports"])
            candidates.append(
                StationCandidate(
                    station_id=station_id,
                    distance_km=float(state.get("distance_km", 2.0)),
                    travel_time_min=float(state.get("travel_time_min", 5.0)),
                    free_ports=available_ports,
                    total_ports=max(total_ports, 1),
                    power_kw=float(state.get("power_kw", 22.0)),
                )
            )
        return candidates

    def recommend_action(self) -> int:
        self.station_candidates = self._build_recommendation_candidates()
        if not self.station_candidates:
            return 0
        vehicle_state = {"soc_pct": 50.0, "battery_kwh": 40.0, "consumption_wh_per_km": 200.0}
        ranked = self.recommendation_engine.recommend(vehicle_state, self.station_candidates, top_k=self.num_stations)
        top_station = ranked[0].station_id
        if top_station in self.station_ids:
            return self.station_ids.index(top_station)
        return 0

    def _build_observation(self) -> dict[str, np.ndarray]:
        available = []
        occupied = []
        price_per_kwh = []

        for station in self.stations:
            station_id = station[0]
            agg = self.repo.get_station_aggregate(station_id)
            state = self.repo.get_station_state(station_id)
            available.append(agg["available_ports"])
            occupied.append(agg["occupied_ports"])
            price_per_kwh.append(float(state["price_per_kwh"] or 0.0))

        return {
            "available_ports": np.array(available, dtype=np.int32),
            "occupied_ports": np.array(occupied, dtype=np.int32),
            "price_per_kwh": np.array(price_per_kwh, dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if hasattr(self.repo, "reset_ports_for_new_simulation"):
            try:
                self.repo.reset_ports_for_new_simulation()
            except Exception:
                pass
        self._active_ports = {}
        self._step_index = 0
        self.station_candidates = self._build_recommendation_candidates()
        self.state = self._build_observation()
        self.done = False
        self.last_selected_station = None
        return self.state, {}

    def _build_recommendation_engine(self) -> None:
        if self.recommendation_engine is not None:
            return
        candidates = []
        for station in self.stations:
            station_id = station[0]
            station_state = self.repo.get_station_state(station_id)
            if not station_state:
                continue
            available = sum(1 for port in station_state["ports"] if port["status"] == "Available")
            total = len(station_state["ports"])
            candidates.append(
                StationCandidate(
                    station_id=station_id,
                    distance_km=2.0,
                    travel_time_min=5.0,
                    free_ports=available,
                    total_ports=total,
                    power_kw=float(station_state.get("price_per_kwh") or 22.0),
                )
            )
        self.recommendation_engine = RecommendationEngine()
        self.recommendation_engine.candidates = candidates

    def _release_one_charging_port(self) -> None:
        cursor = self.connection.execute("SELECT port_id FROM ports WHERE status = 'Charging' ORDER BY status_updated_at LIMIT 1")
        row = cursor.fetchone()
        if row:
            self.repo.log_status_change(row[0], "Available")

    def _acquire_port_for_station(self, station_id: str) -> bool:
        station_state = self.repo.get_station_state(station_id)
        if not station_state:
            return False
        available_ports = [port for port in station_state["ports"] if port["status"] == "Available"]
        if not available_ports:
            return False
        port_id = available_ports[0]["port_id"]
        self.repo.log_status_change(port_id, "Charging")
        return True

    def step(self, action):
        self.station_candidates = self._build_recommendation_candidates()
        self.state = self._build_observation()
        action = int(action)
        station_index = min(max(0, action), len(self.station_ids) - 1)
        self.last_selected_station = self.station_ids[station_index]

        vehicle_state = {"soc_pct": 50.0, "battery_kwh": 40.0, "consumption_wh_per_km": 200.0}
        ranked = self.recommendation_engine.recommend(vehicle_state, self.station_candidates, top_k=self.num_stations)
        top_station = ranked[0].station_id if ranked else None

        reward = float(self.state["available_ports"].sum())
        station_state = self.repo.get_station_state(self.last_selected_station)
        if station_state is not None:
            available_ports = sum(1 for port in station_state["ports"] if port["status"] == "Available")
            occupied_ports = sum(1 for port in station_state["ports"] if port["status"] != "Available")
            price = float(station_state.get("price_per_kwh") or 0.0)
            reward += available_ports * 2.0
            reward -= occupied_ports * 0.5
            reward -= price * 0.1
            if available_ports > 0:
                reward += 6.0
                self._acquire_port_for_station(self.last_selected_station)
            else:
                reward -= 5.0
            if self.last_selected_station == top_station:
                reward += 5.0
            else:
                reward -= 2.0

        if self._step_index > 0 and self._step_index % 3 == 0:
            self._release_one_charging_port()

        self._step_index += 1
        self.done = self._step_index >= self.max_episode_steps
        return self.state, reward, self.done, False, {"selected_station": self.last_selected_station, "recommended": top_station}

    def render(self):
        print("EVChargingEnv observation:")
        for key, value in self.state.items():
            print(f"  {key}: {value.tolist()}")

    def close(self):
        self.connection.close()
