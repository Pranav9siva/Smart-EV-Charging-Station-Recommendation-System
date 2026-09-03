from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation_engine.engine import RecommendationEngine
from src.route_planning.network_graph import NetworkGraph
from src.station_management.station_state import ChargingStation, StationManager


def main() -> None:
    net = NetworkGraph()
    for node in ["A", "B", "C", "D"]:
        net.add_node(node)
    net.add_edge("A", "B", travel_time=2.0, distance=1.0)
    net.add_edge("B", "C", travel_time=2.0, distance=1.0)
    net.add_edge("C", "D", travel_time=2.0, distance=1.0)
    net.add_edge("A", "D", travel_time=5.0, distance=2.0)

    stations = [
        ChargingStation("A", capacity=2, occupancy=0, waiting_time=4.0),
        ChargingStation("C", capacity=2, occupancy=0, waiting_time=1.0),
        ChargingStation("D", capacity=1, occupancy=0, waiting_time=2.0),
    ]
    engine = RecommendationEngine(StationManager(stations), net)
    recommendation = engine.recommend("A", "D")
    print(recommendation.station_id if recommendation else "NONE")


if __name__ == "__main__":
    main()
