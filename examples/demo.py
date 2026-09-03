from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ev_management.ev_state import ElectricVehicle, EVManager
from src.recommendation_engine.engine import RecommendationEngine
from src.route_planning.network_graph import NetworkGraph
from src.station_management.station_state import ChargingStation, StationManager
from src.storage.sqlite_store import SQLiteStore


def run_demo() -> None:
    network = NetworkGraph()
    for node in ["A", "B", "C", "D"]:
        network.add_node(node)
    network.add_edge("A", "B", travel_time=2.0, distance=1.0)
    network.add_edge("B", "C", travel_time=2.0, distance=1.0)
    network.add_edge("C", "D", travel_time=2.0, distance=1.0)
    network.add_edge("A", "D", travel_time=5.0, distance=2.0)

    stations = [
        ChargingStation("A", capacity=2, occupancy=0, waiting_time=4.0),
        ChargingStation("C", capacity=2, occupancy=0, waiting_time=1.0),
        ChargingStation("D", capacity=1, occupancy=0, waiting_time=2.0),
    ]
    station_manager = StationManager(stations)
    engine = RecommendationEngine(station_manager, network)

    ev_manager = EVManager()
    ev_manager.register_vehicle(ElectricVehicle(vehicle_id="ev-1", battery_level=0.43, destination="D"))

    vehicle = ev_manager.get_vehicle("ev-1")
    if vehicle is None:
        raise RuntimeError("Vehicle was not registered")

    recommended_station = engine.recommend("A", vehicle.destination)
    if recommended_station is None:
        raise RuntimeError("No station could be recommended")

    store = SQLiteStore("demo_recommendations.sqlite")
    store.save_recommendation(vehicle.vehicle_id, "A", vehicle.destination, recommended_station.station_id)

    print(f"Recommended station for {vehicle.vehicle_id}: {recommended_station.station_id}")
    print("Stored recommendation in demo_recommendations.sqlite")


if __name__ == "__main__":
    run_demo()
