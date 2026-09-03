"""Project data helpers and paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
STATIONS_DIR = DATA_DIR / "stations"

__all__ = ["ROOT", "DATA_DIR", "STATIONS_DIR"]
