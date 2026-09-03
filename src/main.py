from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ev_management.ev_state import ElectricVehicle, EVManager
from src.recommendation_engine.engine import RecommendationEngine
from src.route_planning.network_graph import NetworkGraph
from src.rl_env.ev_charging_env import EVChargingEnv
from src.rl_env.training import train_and_evaluate_policy, train_and_evaluate_ppo
from src.station_management.station_state import ChargingStation, StationManager
from src.storage.sqlite_store import SQLiteStore


def main() -> None:
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
    station_id = None
    if isinstance(recommended_station, dict):
        station_id = recommended_station.get("station")
    elif recommended_station is not None:
        station_id = getattr(recommended_station, "station_id", None)

    if not station_id:
        print("No available charging station found.")
        return

    print(f"Recommended charging station: {station_id}")

    store = SQLiteStore("recommendations.sqlite")
    store.save_recommendation(vehicle.vehicle_id, "A", vehicle.destination, station_id)

    env = EVChargingEnv(num_stations=3, num_vehicles=4)
    observation, _ = env.reset()
    action = env.action_space.sample()
    _, reward, _, _, _ = env.step(action)
    print(f"Sample env step: action={action}, reward={reward}")

    policy_metrics = train_and_evaluate_policy(episodes=3, steps_per_episode=4)
    print(f"Recommendation baseline summary: {policy_metrics}")

    ppo_metrics = train_and_evaluate_ppo(steps=500, num_stations=3, num_vehicles=4)
    print(f"PPO training evaluation summary: {ppo_metrics}")


if __name__ == "__main__":
    main()
