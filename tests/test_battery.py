from src.ev_model.battery import charge_time_minutes, energy_needed_kwh, remaining_range_km


def test_remaining_range_normal_case() -> None:
    assert remaining_range_km(60.0, 40.0, 200.0) == 100.0


def test_energy_needed_for_above_target_is_zero() -> None:
    assert energy_needed_kwh(80.0, 70.0, 40.0) == 0.0


def test_reserve_buffer_limits_range() -> None:
    assert remaining_range_km(20.0, 50.0, 200.0, reserve_pct=15.0) == 12.5


def test_near_zero_soc_has_small_range() -> None:
    assert remaining_range_km(2.0, 30.0, 250.0) == 0.0


def test_high_power_dc_charge_time_is_short() -> None:
    assert charge_time_minutes(20.0, 150.0) < 20.0
