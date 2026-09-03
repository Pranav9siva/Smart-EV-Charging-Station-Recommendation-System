import sqlite3
import sys
import types
from pathlib import Path

import numpy as np

from src.station_management.manager import StationManager
from src.recommendation_engine.engine import RecommendationEngine
from src.rl_env.gym_ev_charging_env import GymEVChargingEnv
from src.rl_env.traci_ev_charging_env import TraciEVChargingEnv
from src.simulation.controller import SimulationController
import traci


def test_refresh_station_state_updates_db(tmp_path: Path) -> None:
    db_path = tmp_path / "stations.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE stations (station_id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL, operator TEXT, address TEXT, grid_zone_id TEXT, data_source TEXT)"
    )
    conn.execute(
        "CREATE TABLE ports (port_id TEXT PRIMARY KEY, station_id TEXT, connector_type TEXT, power_kw REAL, status TEXT, status_updated_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE occupancy_log (id INTEGER PRIMARY KEY AUTOINCREMENT, port_id TEXT, status TEXT, changed_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE price_history (id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT, price_per_kwh REAL, tariff_period TEXT, recorded_at TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("st1", "Test Station", 12.0, 77.0, "Test", "Test Address", None, "simulated"),
    )
    conn.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("st1_p1", "st1", "CCS", 22.0, "Available", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()

    manager = StationManager(connection=conn)
    env = object.__new__(TraciEVChargingEnv)
    env.station_manager = manager
    env.station_adapter = type("Adapter", (), {})()
    env.station_adapter.list_all_stations = lambda: [{"station_id": "st1"}]
    env.station_adapter.get_station_state = lambda station_id: {"station_id": station_id, "ports": [{"port_id": "st1_p1", "status": "Available"}]}
    env.station_adapter.update_port_status = manager.update_port_status
    env.tracked_vehicle_ids = ["veh1"]
    env._active_ports = {}
    env._step_index = 0

    env._refresh_station_state()

    row = conn.execute("SELECT status FROM ports WHERE port_id = ?", ("st1_p1",)).fetchone()
    assert row is not None
    assert row[0] == "Charging"


def test_charging_station_manager_metrics_and_port_lifecycle(tmp_path: Path) -> None:
    db_path = tmp_path / "stations.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE stations (station_id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL, operator TEXT, address TEXT, grid_zone_id TEXT, data_source TEXT)"
    )
    conn.execute(
        "CREATE TABLE ports (port_id TEXT PRIMARY KEY, station_id TEXT, connector_type TEXT, power_kw REAL, status TEXT, status_updated_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE occupancy_log (id INTEGER PRIMARY KEY AUTOINCREMENT, port_id TEXT, status TEXT, changed_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE price_history (id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT, price_per_kwh REAL, tariff_period TEXT, recorded_at TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("st1", "Test Station", 12.0, 77.0, "Test", "Test Address", None, "simulated"),
    )
    conn.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("st1_p1", "st1", "CCS", 22.0, "Available", "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("st1_p2", "st1", "CCS", 22.0, "Charging", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()

    manager = StationManager(connection=conn)
    metrics = manager.get_station_metrics("st1")
    assert metrics is not None
    assert metrics["available_ports"] == 1
    assert metrics["occupied_ports"] == 1
    assert metrics["total_ports"] == 2
    assert metrics["avg_wait_estimate"] == 11.0 or metrics["avg_wait_estimate"] == 0.0

    port_id = manager.reserve_port("st1")
    assert port_id == "st1_p1"
    reserved_status = conn.execute("SELECT status FROM ports WHERE port_id = ?", (port_id,)).fetchone()[0]
    assert reserved_status == "Charging"

    manager.release_port(port_id)
    released_status = conn.execute("SELECT status FROM ports WHERE port_id = ?", (port_id,)).fetchone()[0]
    assert released_status == "Available"
    conn.close()


def test_gym_env_does_not_close_externally_owned_station_manager(tmp_path: Path) -> None:
    db_path = tmp_path / "stations.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE stations (station_id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL, operator TEXT, address TEXT, grid_zone_id TEXT, data_source TEXT)"
    )
    conn.execute(
        "CREATE TABLE ports (port_id TEXT PRIMARY KEY, station_id TEXT, connector_type TEXT, power_kw REAL, status TEXT, status_updated_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE occupancy_log (id INTEGER PRIMARY KEY AUTOINCREMENT, port_id TEXT, status TEXT, changed_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE price_history (id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT, price_per_kwh REAL, tariff_period TEXT, recorded_at TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("st1", "Test Station", 12.0, 77.0, "Test", "Test Address", None, "simulated"),
    )
    conn.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("st1_p1", "st1", "CCS", 22.0, "Available", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()

    manager = StationManager(connection=conn)
    env = GymEVChargingEnv(db_path=None, station_manager=manager, tracked_vehicle_count=2, candidate_count=2)
    env.close()

    state = manager.get_station_state("st1")
    assert state is not None
    assert manager.reserve_port("st1") == "st1_p1"
    conn.close()


def test_controller_skips_reroute_when_sumo_vehicle_is_not_known(monkeypatch) -> None:
    controller = object.__new__(SimulationController)
    controller.vm_to_sumo = {"veh1": "sumo_veh1"}
    controller.station_manager = type("DummyStationManager", (), {})()
    controller.station_manager.reserve_port = lambda station_id: "p1"
    controller.station_manager.get_station_state = lambda station_id: {"station_id": station_id, "lat": 12.0, "lon": 77.0}
    controller.station_manager._snap_station_to_lane = lambda net, lat, lon, max_snap_distance_m=200.0: ("lane1", 0.0)
    controller.station_manager.repo = type("DummyRepo", (), {"connection": type("DummyConn", (), {"execute": lambda self, *args, **kwargs: None, "commit": lambda self: None})()})()
    controller.vehicle_manager = type("DummyVehicleManager", (), {"get_vehicle": lambda self, vid: type("DummyVehicle", (), {"battery_pct": 15.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 20.0, "destination": "dest"})()})()
    controller.recommender = type("DummyRecommender", (), {"recommend": lambda self, vehicle_state, model_path=None: {"station": "st1", "waiting_time": 2.0, "queue_length": 1, "charging_cost": 0.2, "grid_load": 10.0, "available_ports": 2}})()
    controller.net = type("DummyNet", (), {"getLane": lambda self, lane_id: type("DummyLane", (), {"getEdge": lambda self: type("DummyEdge", (), {"getID": lambda self: "edge_1"})()})()})()
    controller.assignments = {}
    controller.recommendation_log = []
    controller.reward_curve = []
    controller.charging_events = []
    controller.model = None
    controller.model_path = ""
    controller.tracked = 1
    controller.candidate_count = 1
    controller.rec_call_count = 0
    controller.low_battery_count = 0

    called = {"count": 0}

    def fake_change_target(vid, target):
        called["count"] += 1

    monkeypatch.setattr(traci.vehicle, "changeTarget", fake_change_target)
    monkeypatch.setattr(traci.vehicle, "getIDList", lambda: [])

    controller._handle_low_battery("veh1", type("DummyVehicleRecord", (), {"battery_pct": 15.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 20.0, "destination": "dest"})(), 0)

    assert called["count"] == 0


def test_controller_handles_padded_observation_for_ppo_reward(monkeypatch) -> None:
    controller = object.__new__(SimulationController)
    controller.vm_to_sumo = {"veh1": "sumo_veh1"}
    controller.station_manager = type("DummyStationManager", (), {})()
    controller.station_manager.reserve_port = lambda station_id: "p1"
    controller.station_manager.get_station_state = lambda station_id: {"station_id": station_id, "lat": 12.0, "lon": 77.0}
    controller.station_manager._snap_station_to_lane = lambda net, lat, lon, max_snap_distance_m=200.0: ("lane1", 0.0)
    controller.station_manager.repo = type("DummyRepo", (), {"connection": type("DummyConn", (), {"execute": lambda self, *args, **kwargs: None, "commit": lambda self: None})()})()
    controller.vehicle_manager = type("DummyVehicleManager", (), {"get_vehicle": lambda self, vid: type("DummyVehicle", (), {"battery_pct": 15.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 20.0, "destination": "dest"})()})()
    controller.recommender = type("DummyRecommender", (), {"recommend": lambda self, vehicle_state, model_path=None: {"station": "st1", "waiting_time": 2.0, "queue_length": 1, "charging_cost": 0.2, "grid_load": 10.0, "available_ports": 2, "stations": [{"station_id": "st1"}]}})()
    controller.net = type("DummyNet", (), {"getLane": lambda self, lane_id: type("DummyLane", (), {"getEdge": lambda self: type("DummyEdge", (), {"getID": lambda self: "edge_1"})()})()})()
    controller.assignments = {}
    controller.recommendation_log = []
    controller.reward_curve = []
    controller.charging_events = []
    controller.model = type("DummyModel", (), {"predict": lambda self, obs, deterministic=True: (0, None)})()
    controller.model_path = ""
    controller.tracked = 1
    controller.candidate_count = 1
    controller.rec_call_count = 0
    controller.low_battery_count = 0

    class DummyEnv:
        def reset(self):
            return ({"vehicles": np.zeros((2, 6), dtype=np.float32), "stations": np.zeros((1, 6), dtype=np.float32)}, {})

        def step(self, action):
            return ({}, 1.0, False, False, {})

        def close(self):
            return None

    monkeypatch.setattr("src.simulation.controller.np", __import__("numpy"))
    dummy_module = types.ModuleType("src.rl_env.gym_ev_charging_env")
    dummy_module.GymEVChargingEnv = lambda *args, **kwargs: DummyEnv()
    monkeypatch.setitem(sys.modules, "src.rl_env.gym_ev_charging_env", dummy_module)
    monkeypatch.setattr(traci.vehicle, "getIDList", lambda: ["sumo_veh1"])
    monkeypatch.setattr(traci.edge, "getIDList", lambda: ["edge_1"])

    controller._handle_low_battery("veh1", type("DummyVehicleRecord", (), {"battery_pct": 15.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 20.0, "destination": "dest"})(), 0)

    assert controller.reward_curve[-1] == 1.0


def test_controller_uses_sumo_vehicle_id_for_reroute(monkeypatch) -> None:
    controller = object.__new__(SimulationController)
    controller.vm_to_sumo = {"veh1": "sumo_veh1"}
    controller.station_manager = type("DummyStationManager", (), {})()
    controller.station_manager.reserve_port = lambda station_id: "p1"
    controller.station_manager.get_station_state = lambda station_id: {"station_id": station_id, "lat": 12.0, "lon": 77.0}
    controller.station_manager._snap_station_to_lane = lambda net, lat, lon, max_snap_distance_m=200.0: ("lane1", 0.0)
    controller.station_manager.repo = type("DummyRepo", (), {"connection": type("DummyConn", (), {"execute": lambda self, *args, **kwargs: None, "commit": lambda self: None})()})()
    controller.vehicle_manager = type("DummyVehicleManager", (), {"get_vehicle": lambda self, vid: type("DummyVehicle", (), {"battery_pct": 15.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 20.0, "destination": "dest"})()})()
    controller.recommender = type("DummyRecommender", (), {"recommend": lambda self, vehicle_state, model_path=None: {"station": "st1", "waiting_time": 2.0, "queue_length": 1, "charging_cost": 0.2, "grid_load": 10.0, "available_ports": 2}})()
    controller.net = type("DummyNet", (), {"getLane": lambda self, lane_id: type("DummyLane", (), {"getEdge": lambda self: type("DummyEdge", (), {"getID": lambda self: "edge_1"})()})()})()
    controller.assignments = {}
    controller.recommendation_log = []
    controller.reward_curve = []
    controller.charging_events = []
    controller.model = None
    controller.model_path = ""
    controller.tracked = 1
    controller.candidate_count = 1
    controller.rec_call_count = 0
    controller.low_battery_count = 0

    captured = {}

    def fake_change_target(vid, target):
        captured["vid"] = vid
        captured["target"] = target

    monkeypatch.setattr(traci.vehicle, "changeTarget", fake_change_target)
    monkeypatch.setattr(traci.vehicle, "getIDList", lambda: ["sumo_veh1"])
    monkeypatch.setattr(traci.edge, "getIDList", lambda: ["edge_1"])

    controller._handle_low_battery("veh1", type("DummyVehicleRecord", (), {"battery_pct": 15.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 20.0, "destination": "dest"})(), 0)

    assert captured["vid"] == "sumo_veh1"
    assert captured["target"] == "edge_1"


def test_run_starts_sumo_when_traci_is_disconnected(monkeypatch) -> None:
    controller = object.__new__(SimulationController)
    controller.start = lambda: None
    controller.stop = lambda: None
    controller._save_results = lambda: None
    controller._update_vehicles = lambda step: None
    controller._process_charging = lambda step: None
    controller._refresh_dashboard = lambda: None
    controller._collect_metrics = lambda step: None
    controller.fleet_size = 1
    controller.tracked = 1
    controller.vehicle_manager = type("DummyVehicleManager", (), {"list_vehicles": lambda self: [], "generate_fleet": lambda self, **kwargs: None})()
    controller.simulation_metrics = []
    controller.recommendation_log = []
    controller.charging_events = []
    controller.assignments = {}
    controller.reward_curve = []
    controller.low_battery_count = 0
    controller.rec_call_count = 0
    controller.injected_count = 0

    start_calls = []

    def fake_start() -> None:
        start_calls.append("start")

    controller.start = fake_start
    monkeypatch.setattr(traci, "__name__", "traci")
    monkeypatch.setattr(traci, "isConnected", lambda: False, raising=False)
    monkeypatch.setattr(traci, "simulationStep", lambda: None, raising=False)

    controller.run(steps=1)

    assert start_calls == ["start"]


def test_process_charging_starts_when_target_edge_is_reached(monkeypatch) -> None:
    controller = object.__new__(SimulationController)
    controller.assignments = {}
    controller.charging_events = []
    controller.vm_to_sumo = {"veh1": "veh1"}
    controller.station_manager = type("DummyStationManager", (), {})()
    controller.station_manager.get_station_state = lambda station_id: {"station_id": station_id, "ports": [{"port_id": "p1", "power_kw": 22.0, "status": "Charging"}]}
    controller.vehicle_manager = type("DummyVehicleManager", (), {"get_vehicle": lambda self, vid: type("DummyVehicle", (), {"battery_pct": 20.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 40.0, "destination": "dest"})()})()

    assignment = type("DummyAssignment", (), {"station_id": "st1", "port_id": "p1", "status": "enroute", "charging_end_step": None, "target_edge_id": "station_edge"})()
    controller.assignments["veh1"] = assignment

    monkeypatch.setattr(traci.vehicle, "getRoadID", lambda vid: "station_edge")
    monkeypatch.setattr(traci.vehicle, "getIDList", lambda: ["veh1"])

    controller._process_charging(0)

    assert assignment.status == "charging"
    assert controller.charging_events[0]["vehicle_id"] == "veh1"


def test_generate_sumo_additional_file_with_mock_sumolib(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "stations.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE stations (station_id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL, operator TEXT, address TEXT, grid_zone_id TEXT, data_source TEXT)"
    )
    conn.execute(
        "CREATE TABLE ports (port_id TEXT PRIMARY KEY, station_id TEXT, connector_type TEXT, power_kw REAL, status TEXT, status_updated_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE occupancy_log (id INTEGER PRIMARY KEY AUTOINCREMENT, port_id TEXT, status TEXT, changed_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE price_history (id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT, price_per_kwh REAL, tariff_period TEXT, recorded_at TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("st1", "Test Station", 12.0, 77.0, "Test", "Test Address", None, "simulated"),
    )
    conn.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("st1_p1", "st1", "CCS", 22.0, "Available", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()

    class DummyLane:
        def __init__(self, lane_id: str, length: float):
            self._id = lane_id
            self._length = length

        def getID(self) -> str:
            return self._id

        def getLength(self) -> float:
            return self._length

        def getEdge(self):
            class Edge:
                def __init__(self, edge_id: str):
                    self._id = edge_id

                def getID(self) -> str:
                    return self._id

            return Edge(self._id.split("_")[0])

    class DummyNet:
        def convertLonLat2XY(self, lon, lat):
            return lon, lat

        def getNeighboringLanes(self, x, y, max_dist):
            return [("edge0_0", 10.0)]

        def getLane(self, lane_id):
            return DummyLane(lane_id, 50.0)

    fake_sumolib = type("FakeSumoLib", (), {})()
    fake_sumolib.net = type("NetModule", (), {"readNet": lambda self, path: DummyNet()})()
    monkeypatch.setitem(sys.modules, "sumolib", fake_sumolib)

    manager = StationManager(connection=conn)
    output_file = tmp_path / "stations.add.xml"
    stations = manager.generate_sumo_additional_file(net_path="/tmp/fake.net.xml", output_path=output_file)
    assert stations
    content = output_file.read_text(encoding="utf-8")
    assert "<chargingStation" in content
    assert "id=\"st1\"" in content
    conn.close()


def test_controller_refresh_uses_single_station_snapshot_query(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "stations.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE stations (station_id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL, operator TEXT, address TEXT, grid_zone_id TEXT, data_source TEXT)"
    )
    conn.execute(
        "CREATE TABLE ports (port_id TEXT PRIMARY KEY, station_id TEXT, connector_type TEXT, power_kw REAL, status TEXT, status_updated_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE occupancy_log (id INTEGER PRIMARY KEY AUTOINCREMENT, port_id TEXT, status TEXT, changed_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE price_history (id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT, price_per_kwh REAL, tariff_period TEXT, recorded_at TIMESTAMP)"
    )
    for index in range(1, 6):
        station_id = f"st{index}"
        conn.execute(
            "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (station_id, f"Station {index}", 12.0 + index * 0.01, 77.0 + index * 0.01, "Test", "Addr", "zone", "simulated"),
        )
        conn.execute(
            "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (f"{station_id}_p1", station_id, "CCS", 22.0 + index, "Available", "2026-01-01T00:00:00+00:00"),
        )
    conn.commit()

    manager = StationManager(connection=conn)
    repo_call_count = {"count": 0}
    original_get_station_snapshots = manager.repo.get_station_snapshots

    def counted_get_station_snapshots():
        repo_call_count["count"] += 1
        return original_get_station_snapshots()

    manager.repo.get_station_snapshots = counted_get_station_snapshots  # type: ignore[method-assign]
    manager.repo.get_station_state = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("refresh should not query per-station state"))  # type: ignore[assignment]
    manager.repo.get_station_aggregate = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("refresh should not query per-station aggregate"))  # type: ignore[assignment]

    controller = object.__new__(SimulationController)
    controller.station_manager = manager
    controller.vehicle_manager = type("DummyVehicleManager", (), {"list_vehicles": lambda self: [], "list_tracked_vehicles": lambda self: []})()
    controller.assignments = {}
    controller.recommendation_log = []
    controller.charging_events = []
    controller.simulation_metrics = []
    controller.dashboard = type("DummyDashboard", (), {"update": lambda self, state: None})()
    controller.vm_to_sumo = {}
    controller.net = None
    controller.current_step = 1
    controller._station_snapshot_cache = []
    controller._station_snapshot_cache_by_id = {}
    controller._station_snapshot_cache_step = None

    monkeypatch.setattr(traci.vehicle, "getIDList", lambda: [], raising=False)

    controller._refresh_dashboard()

    assert repo_call_count["count"] == 1
    conn.close()


def test_controller_reuses_loaded_ppo_model_across_multiple_recommendations(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "stations.sqlite"
    model_path = tmp_path / "ppo_ev_final.zip"
    model_path.write_bytes(b"dummy model file")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE stations (station_id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL, operator TEXT, address TEXT, grid_zone_id TEXT, data_source TEXT)"
    )
    conn.execute(
        "CREATE TABLE ports (port_id TEXT PRIMARY KEY, station_id TEXT, connector_type TEXT, power_kw REAL, status TEXT, status_updated_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE occupancy_log (id INTEGER PRIMARY KEY AUTOINCREMENT, port_id TEXT, status TEXT, changed_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE price_history (id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT, price_per_kwh REAL, tariff_period TEXT, recorded_at TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("st1", "Station 1", 12.0, 77.0, "Test", "Addr", "zone", "simulated"),
    )
    conn.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("st1_p1", "st1", "CCS", 22.0, "Available", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()

    manager = StationManager(connection=conn)

    load_calls = {"count": 0}

    class DummyModel:
        def predict(self, obs, deterministic=True):
            return 0, None

    def fake_load(path):
        load_calls["count"] += 1
        return DummyModel()

    class DummyEnv:
        vehicle_feat_dim = 6

        def __init__(self, *args, **kwargs):
            self.current_candidates = [
                {
                    "station_id": "st1",
                    "distance_km": 1.0,
                    "travel_time_min": 2.0,
                    "avg_wait_min": 1.0,
                    "queue_len": 0,
                    "free_ports": 1,
                    "price_per_kwh": 0.2,
                    "grid_load_kw": 0.0,
                }
            ]

        def reset(self):
            return ({"vehicles": np.zeros((1, 6), dtype=np.float32), "stations": np.zeros((1, 6), dtype=np.float32)}, {})

        def step(self, action):
            return ({}, 1.0, False, False, {})

        def close(self):
            return None

    dummy_module = types.ModuleType("src.rl_env.gym_ev_charging_env")
    dummy_module.GymEVChargingEnv = DummyEnv
    monkeypatch.setitem(sys.modules, "src.rl_env.gym_ev_charging_env", dummy_module)
    monkeypatch.setattr("src.recommendation_engine.engine.PPO.load", fake_load, raising=False)

    controller = object.__new__(SimulationController)
    controller.vm_to_sumo = {"veh1": "sumo_veh1", "veh2": "sumo_veh2"}
    controller.station_manager = manager
    controller.vehicle_manager = type(
        "DummyVehicleManager",
        (),
        {
            "get_vehicle": lambda self, vid: type(
                "DummyVehicle",
                (),
                {"battery_pct": 15.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 20.0, "destination": "dest"},
            )(),
        },
    )()
    controller.recommender = RecommendationEngine(manager, model_path=str(model_path))
    controller.net = type("DummyNet", (), {"getEdges": lambda self: [], "convertLonLat2XY": lambda self, lon, lat: (lon, lat)})()
    controller.assignments = {}
    controller.recommendation_log = []
    controller.reward_curve = []
    controller.charging_events = []
    controller.model = None
    controller.model_path = str(model_path)
    controller.tracked = 1
    controller.candidate_count = 1
    controller.rec_call_count = 0
    controller.low_battery_count = 0
    controller.last_explanation = None

    monkeypatch.setattr(traci.vehicle, "changeTarget", lambda vid, target: None)
    monkeypatch.setattr(traci.vehicle, "getIDList", lambda: ["sumo_veh1", "sumo_veh2"])
    monkeypatch.setattr(traci.vehicle, "getRoute", lambda vid: ["edge_1"], raising=False)
    monkeypatch.setattr(traci.edge, "getIDList", lambda: ["edge_1"])

    controller._handle_low_battery("veh1", type("DummyVehicleRecord", (), {"battery_pct": 15.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 20.0, "destination": "dest"})(), 0)
    controller._handle_low_battery("veh2", type("DummyVehicleRecord", (), {"battery_pct": 14.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 18.0, "destination": "dest"})(), 1)

    assert load_calls["count"] == 1
    assert controller.recommender.model_load_count == 1
    assert len(controller.recommendation_log) == 2
    assert controller.recommendation_log[-1]["selected_station"] == "st1"


def test_dashboard_and_metrics_share_current_station_snapshot(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "stations.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE stations (station_id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL, operator TEXT, address TEXT, grid_zone_id TEXT, data_source TEXT)"
    )
    conn.execute(
        "CREATE TABLE ports (port_id TEXT PRIMARY KEY, station_id TEXT, connector_type TEXT, power_kw REAL, status TEXT, status_updated_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE occupancy_log (id INTEGER PRIMARY KEY AUTOINCREMENT, port_id TEXT, status TEXT, changed_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE price_history (id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT, price_per_kwh REAL, tariff_period TEXT, recorded_at TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("st1", "Station 1", 12.0, 77.0, "Test", "Addr", "zone", "simulated"),
    )
    conn.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("st1_p1", "st1", "CCS", 22.0, "Available", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()

    manager = StationManager(connection=conn)
    repo_call_count = {"count": 0}
    original_get_station_snapshots = manager.repo.get_station_snapshots

    def counted_get_station_snapshots():
        repo_call_count["count"] += 1
        return original_get_station_snapshots()

    manager.repo.get_station_snapshots = counted_get_station_snapshots  # type: ignore[method-assign]

    controller = object.__new__(SimulationController)
    controller.station_manager = manager
    controller.vehicle_manager = type("DummyVehicleManager", (), {"list_vehicles": lambda self: [], "list_tracked_vehicles": lambda self: []})()
    controller.assignments = {}
    controller.recommendation_log = []
    controller.charging_events = []
    controller.simulation_metrics = []
    controller.dashboard = type("DummyDashboard", (), {"update": lambda self, state: None})()
    controller.vm_to_sumo = {}
    controller.net = None
    controller.current_step = 1
    controller._station_snapshot_cache = []
    controller._station_snapshot_cache_by_id = {}
    controller._station_snapshot_cache_step = None

    monkeypatch.setattr(traci.vehicle, "getIDList", lambda: [], raising=False)

    controller._refresh_dashboard()
    controller._collect_metrics(1)

    assert repo_call_count["count"] == 1
    conn.close()


def test_low_battery_recommendation_uses_current_available_ports(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "stations.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE stations (station_id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL, operator TEXT, address TEXT, grid_zone_id TEXT, data_source TEXT)"
    )
    conn.execute(
        "CREATE TABLE ports (port_id TEXT PRIMARY KEY, station_id TEXT, connector_type TEXT, power_kw REAL, status TEXT, status_updated_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE occupancy_log (id INTEGER PRIMARY KEY AUTOINCREMENT, port_id TEXT, status TEXT, changed_at TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE price_history (id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT, price_per_kwh REAL, tariff_period TEXT, recorded_at TIMESTAMP)"
    )
    conn.execute(
        "INSERT INTO stations (station_id, name, lat, lon, operator, address, grid_zone_id, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("st1", "Station 1", 12.0, 77.0, "Test", "Addr", "zone", "simulated"),
    )
    conn.execute(
        "INSERT INTO ports (port_id, station_id, connector_type, power_kw, status, status_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("st1_p1", "st1", "CCS", 22.0, "Available", "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()

    manager = StationManager(connection=conn)
    # Prime the cache, then mutate the DB through the manager so the next lookup must reflect the change.
    assert manager.get_station_metrics("st1")["available_ports"] == 1
    assert manager.reserve_port("st1") == "st1_p1"

    controller = object.__new__(SimulationController)
    controller.vm_to_sumo = {"veh1": "sumo_veh1"}
    controller.station_manager = manager
    controller.vehicle_manager = type("DummyVehicleManager", (), {"get_vehicle": lambda self, vid: type("DummyVehicle", (), {"battery_pct": 15.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 20.0, "destination": "dest"})()})()
    controller.recommender = type(
        "DummyRecommender",
        (),
        {
            "recommend": lambda self, vehicle_state, model_path=None: {
                "station": "st1",
                "waiting_time": 2.0,
                "queue_length": 1,
                "charging_cost": 0.2,
                "grid_load": 10.0,
                "available_ports": manager.get_station_metrics("st1")["available_ports"],
            }
        },
    )()
    controller.net = None
    controller.assignments = {}
    controller.recommendation_log = []
    controller.reward_curve = []
    controller.charging_events = []
    controller.model = None
    controller.model_path = ""
    controller.tracked = 1
    controller.candidate_count = 1
    controller.rec_call_count = 0
    controller.low_battery_count = 0
    controller.current_step = 0
    controller._station_snapshot_cache = []
    controller._station_snapshot_cache_by_id = {}
    controller._station_snapshot_cache_step = None
    controller.explainer = type(
        "DummyExplainer",
        (),
        {
            "explain_recommendation": lambda self, **kwargs: {"summary": "ok"},
            "export_explanation_json": lambda self, explanation, path: None,
        },
    )()

    monkeypatch.setattr(traci.vehicle, "changeTarget", lambda vid, target: None)
    monkeypatch.setattr(traci.vehicle, "getIDList", lambda: ["sumo_veh1"], raising=False)

    controller._handle_low_battery("veh1", type("DummyVehicleRecord", (), {"battery_pct": 15.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 20.0, "destination": "dest"})(), 0)

    selected = controller.recommendation_log[-1]
    assert selected["available_ports"] == 0
    conn.close()
