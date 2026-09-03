from src.recommendation_engine.engine import RecommendationEngine


def test_recommendation_reason_contains_selected_station_factors() -> None:
    engine = RecommendationEngine(object())
    reason = engine._build_recommendation_reason(
        {
            "distance_km": 2.4,
            "price_per_kwh": 4.27,
            "avg_wait_min": 0.5,
            "free_ports": 2,
            "normalized": {"distance": 0.1, "wait": 0.05, "cost": 0.2},
        },
        {"battery_pct": 18.0},
    )

    assert "low wait" in reason
    assert "available port" in reason
    assert "shorter distance" in reason
    assert "2.4 km" in reason
    assert "4.27/kWh" in reason
    assert "SOC 18.0%" in reason