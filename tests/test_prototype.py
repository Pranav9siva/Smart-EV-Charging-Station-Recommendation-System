from src.ev_management.ev_state import ElectricVehicle, EVManager
from src.recommendation_engine.engine import RecommendationEngine
from src.route_planning.network_graph import NetworkGraph
from src.station_management.station_state import ChargingStation, StationManager
from src.storage.sqlite_store import SQLiteStore
from src.ev_management.vehicle_manager import VehicleManager


def test_ev_manager_tracks_vehicle_state() -> None:
    manager = EVManager()
    vehicle = ElectricVehicle(vehicle_id="ev-1", battery_level=0.42, destination="C")
    manager.register_vehicle(vehicle)

    fetched = manager.get_vehicle("ev-1")

    assert fetched is not None
    assert fetched.battery_level == 0.42

    manager.update_vehicle_state("ev-1", battery_level=0.81)
    assert manager.get_vehicle("ev-1").battery_level == 0.81


def test_recommendation_prefers_route_station_when_available() -> None:
    network = NetworkGraph()
    network.add_node("A")
    network.add_node("B")
    network.add_node("C")
    network.add_edge("A", "B", travel_time=2.0, distance=1.0)
    network.add_edge("B", "C", travel_time=2.0, distance=1.0)

    stations = [
        ChargingStation("A", capacity=2, occupancy=0, waiting_time=3.0),
        ChargingStation("C", capacity=2, occupancy=0, waiting_time=1.0),
    ]
    manager = StationManager(stations)
    engine = RecommendationEngine(manager, network)

    result = engine.recommend("A", "C")

    assert result is not None
    assert result.station_id == "C"


def test_sqlite_store_persists_recommendation_records(tmp_path) -> None:
    db_path = tmp_path / "recommendations.sqlite"
    store = SQLiteStore(db_path)

    store.save_recommendation("ev-1", "A", "C", "B")
    rows = store.list_recommendations()

    assert len(rows) == 1
    assert rows[0]["vehicle_id"] == "ev-1"
    assert rows[0]["station_id"] == "B"


def test_vehicle_manager_generates_1000_vehicles_and_tracks_10() -> None:
    manager = VehicleManager()
    vehicles = manager.generate_fleet(count=1000, detail_count=10)

    assert len(vehicles) == 1000
    assert len(manager.list_tracked_vehicles()) == 10

    for vehicle in vehicles[:10]:
        assert 5 <= vehicle.battery_pct <= 95
        assert vehicle.remaining_range_km >= 0.0
        assert vehicle.charging_requirement_kwh >= 0.0
