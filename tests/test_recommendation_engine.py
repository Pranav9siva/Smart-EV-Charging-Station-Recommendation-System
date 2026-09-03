from src.recommendation.engine import RecommendationEngine, StationCandidate


def test_unreachable_station_never_outranks_reachable_one() -> None:
    engine = RecommendationEngine()
    vehicle_state = {"soc_pct": 20.0, "battery_kwh": 40.0, "consumption_wh_per_km": 180.0}
    reachable = StationCandidate(station_id="reachable", distance_km=2.0, travel_time_min=5.0, free_ports=1, total_ports=2, power_kw=22.0)
    unreachable = StationCandidate(station_id="unreachable", distance_km=40.0, travel_time_min=35.0, free_ports=1, total_ports=2, power_kw=22.0)

    reachable_result = engine.score_station(vehicle_state, reachable)
    unreachable_result = engine.score_station(vehicle_state, unreachable)

    assert reachable_result.reachable is True
    assert unreachable_result.reachable is False
    assert reachable_result.score < unreachable_result.score


def test_occupied_station_scores_worse_than_free_one() -> None:
    engine = RecommendationEngine()
    vehicle_state = {"soc_pct": 70.0, "battery_kwh": 40.0, "consumption_wh_per_km": 180.0}
    free = StationCandidate(station_id="free", distance_km=3.0, travel_time_min=6.0, free_ports=2, total_ports=2, power_kw=22.0)
    occupied = StationCandidate(station_id="occupied", distance_km=3.0, travel_time_min=6.0, free_ports=0, total_ports=2, power_kw=22.0)

    free_result = engine.score_station(vehicle_state, free)
    occupied_result = engine.score_station(vehicle_state, occupied)

    assert free_result.score < occupied_result.score


def test_time_weight_changes_ranking_predictably() -> None:
    engine = RecommendationEngine(weights={"time": 0.0, "distance": 0.2, "availability": 0.3, "battery_safety": 0.5})
    vehicle_state = {"soc_pct": 60.0, "battery_kwh": 40.0, "consumption_wh_per_km": 180.0}
    fast = StationCandidate(station_id="fast", distance_km=15.0, travel_time_min=10.0, free_ports=1, total_ports=2, power_kw=50.0)
    slow = StationCandidate(station_id="slow", distance_km=2.0, travel_time_min=20.0, free_ports=1, total_ports=2, power_kw=22.0)

    fast_result = engine.score_station(vehicle_state, fast)
    slow_result = engine.score_station(vehicle_state, slow)

    assert fast_result.score > slow_result.score
