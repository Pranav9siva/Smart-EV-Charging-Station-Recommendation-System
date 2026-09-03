"""Compatibility package for station-management entry points."""

from src.station_management.manager import ChargingStationManager
from src.station_management.station_state import ChargingStation, StationManager

__all__ = ["ChargingStationManager", "ChargingStation", "StationManager"]
