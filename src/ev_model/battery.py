from __future__ import annotations


def remaining_range_km(
    soc_pct: float,
    battery_kwh: float,
    consumption_wh_per_km: float,
    reserve_pct: float = 10.0,
) -> float:
    """Usable range before hitting the reserve buffer."""
    usable_soc = max(0.0, min(100.0, soc_pct - reserve_pct))
    usable_energy_kwh = (usable_soc / 100.0) * battery_kwh
    return max(0.0, usable_energy_kwh * 1000.0 / consumption_wh_per_km)


def energy_needed_kwh(current_soc_pct: float, target_soc_pct: float, battery_kwh: float) -> float:
    """Energy required to go from current_soc_pct to target_soc_pct."""
    if target_soc_pct <= current_soc_pct:
        return 0.0
    return ((target_soc_pct - current_soc_pct) / 100.0) * battery_kwh


def charge_time_minutes(energy_needed_kwh: float, station_power_kw: float, charging_efficiency: float = 0.9) -> float:
    """Estimated wall-clock charging time, accounting for efficiency losses.

    This is a simplified model and ignores tapering above 80% SoC.
    """
    if energy_needed_kwh <= 0.0 or station_power_kw <= 0.0:
        return 0.0
    return (energy_needed_kwh / charging_efficiency) / (station_power_kw / 60.0)
