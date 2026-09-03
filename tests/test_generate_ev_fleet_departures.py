from scripts.generate_ev_fleet import _build_departure_schedule, _compute_departure_stats


def test_departure_schedule_not_compressed_to_one_step() -> None:
    departure_steps = _build_departure_schedule(vehicle_count=1000, depart_window=120)

    assert len(departure_steps) == 1000
    assert min(departure_steps) == 0
    assert max(departure_steps) == 120
    assert len(set(departure_steps)) > 1

    stats = _compute_departure_stats(departure_steps, bucket_size=10)
    assert stats["first_departure"] == 0
    assert stats["last_departure"] == 120
    assert stats["max_vehicles_departing_in_one_step"] < 1000


def test_departure_schedule_is_deterministic() -> None:
    first = _build_departure_schedule(vehicle_count=1000, depart_window=120)
    second = _build_departure_schedule(vehicle_count=1000, depart_window=120)

    assert first == second
