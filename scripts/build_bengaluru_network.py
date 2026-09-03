from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OSM_PATHS = [
    ROOT / "sumo" / "network" / "bangalore_subarea.osm",
    ROOT / "data" / "osm" / "bangalore_subarea.osm",
]
NET_PATH = ROOT / "sumo" / "network" / "bengaluru.net.xml"
FETCH_SCRIPT = ROOT / "scripts" / "fetch_osm_network.py"


def find_osm_file() -> Path | None:
    for path in OSM_PATHS:
        if path.exists():
            return path
    return None


def main() -> None:
    osm_path = find_osm_file()
    if not osm_path:
        print("OSM extract not found. Fetching via scripts/fetch_osm_network.py...")
        subprocess.run(["python", str(FETCH_SCRIPT)], check=True)
        osm_path = find_osm_file()

    if not osm_path:
        raise FileNotFoundError(
            "OSM file still missing after fetch. Please manually export from https://www.openstreetmap.org/export "
            "using bbox 77.590,12.930,77.650,12.990 and save it as either data/osm/bangalore_subarea.osm "
            "or sumo/network/bangalore_subarea.osm."
        )

    target_path = ROOT / "sumo" / "network" / "bangalore_subarea.osm"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if osm_path != target_path:
        target_path.write_bytes(osm_path.read_bytes())
        osm_path = target_path

    print(f"Converting OSM to SUMO net: {osm_path} -> {NET_PATH}")
    subprocess.run(["netconvert", "--osm-files", str(osm_path), "-o", str(NET_PATH)], check=True)
    print(f"SUMO network generated at {NET_PATH}")


if __name__ == "__main__":
    main()
