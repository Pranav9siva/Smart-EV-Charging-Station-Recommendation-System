from src.recommendation_engine.engine import RecommendationEngine
from src.route_planning.network_graph import NetworkGraph
from src.station_management.station_state import ChargingStation, StationManager


def test_recommendation_engine_returns_station() -> None:
    network = NetworkGraph()
    network.add_node("A")
    network.add_node("B")
    network.add_edge("A", "B", travel_time=1.0, distance=1.0)

    stations = [ChargingStation("A", capacity=2, occupancy=1, waiting_time=3.0)]
    manager = StationManager(stations)
    engine = RecommendationEngine(manager, network)

    result = engine.recommend("A", "B")

    assert result is not None
    assert result.station_id == "A"
