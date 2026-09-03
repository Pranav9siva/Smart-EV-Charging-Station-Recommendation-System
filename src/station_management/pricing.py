from __future__ import annotations

import random
from datetime import datetime, time
from pathlib import Path
from sqlite3 import Connection, connect

SOLAR_START = time(hour=9)
SOLAR_END = time(hour=16)
BASE_RATES = {
    "solar_hour": {"Type2": 3.9, "CCS2": 12.5, "CHAdeMO": 12.5, "GB-T": 12.5},
    "non_solar_hour": {"Type2": 4.2, "CCS2": 13.5, "CHAdeMO": 13.5, "GB-T": 13.5},
}


def current_tariff_period(now: datetime | None = None) -> str:
    now = now or datetime.now()
    if SOLAR_START <= now.time() <= SOLAR_END:
        return "solar_hour"
    return "non_solar_hour"


def tariff_rate(connector_type: str, now: datetime | None = None) -> float:
    period = current_tariff_period(now)
    base = BASE_RATES.get(period, {}).get(connector_type, 4.0)
    markup = random.uniform(0.9, 1.1)
    return round(base * markup, 2)


class PricingManager:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.connection = connect(self.db_path)

    def record_price(self, station_id: str, price_per_kwh: float, tariff_period: str) -> None:
        self.connection.execute(
            "INSERT INTO price_history (station_id, price_per_kwh, tariff_period, recorded_at) VALUES (?, ?, ?, datetime('now'))",
            (station_id, price_per_kwh, tariff_period),
        )
        self.connection.commit()

    def update_prices(self) -> None:
        from .station_repository import StationRepository

        repo = StationRepository(self.connection)
        stations = repo.get_all_stations()
        period = current_tariff_period()
        for station in stations:
            price = tariff_rate(station[0])
            self.record_price(station[0], price, period)

    def get_latest_price(self, station_id: str) -> float | None:
        row = self.connection.execute(
            "SELECT price_per_kwh FROM price_history WHERE station_id = ? ORDER BY recorded_at DESC LIMIT 1",
            (station_id,),
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.connection.close()
