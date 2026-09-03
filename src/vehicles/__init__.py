"""Compatibility package for vehicle-management entry points."""

from src.ev_management.ev_state import EVManager, ElectricVehicle
from src.ev_management.vehicle_manager import VehicleManager, VehicleRecord

__all__ = ["EVManager", "ElectricVehicle", "VehicleManager", "VehicleRecord"]
