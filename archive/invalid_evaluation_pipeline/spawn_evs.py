"""Spawn many EVs with validated routes into simulations/bangalore/evs.rou.xml

Generates 1000 vehicles by default and writes a fleet manifest.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import sumolib

ROOT = Path(__file__).resolve().parents[1]
NETWORK = ROOT / "sumo" / "network" / "bengaluru.net.xml"
OUT_ROUTES = ROOT / "simulations" / "bangalore" / "evs.rou.xml"
MANIFEST = ROOT / "data" / "fleet" / "fleet_manifest.json"


def build_many_routes(count: int = 1000, max_retries: int = 20) -> None:
    if not NETWORK.exists():
        raise FileNotFoundError(f"SUMO network not found at {NETWORK}")

    net = sumolib.net.readNet(NETWORK)
    edges = [edge for edge in net.getEdges() if edge.allows("passenger")]
    if len(edges) < 2:
        raise RuntimeError("Need at least 2 passenger edges to generate routes")

    random.seed(11)
    vehicle_specs: list[dict[str, Any]] = []
    for idx in range(1, count + 1):
        # pick origin/destination with retries to ensure a valid path
        origin = random.choice(edges)
        destination = random.choice(edges)
        tries = 0
        path_edges = []
        while origin == destination and tries < max_retries:
            destination = random.choice(edges)
            tries += 1

        tries = 0
        while tries < max_retries:
            try:
                path = net.getShortestPath(origin, destination)
                if path:
                    path_edges = [e.getID() for e in path]
                    break
            except Exception:
                pass
            # pick new destination and retry
            destination = random.choice(edges)
            tries += 1

        if not path_edges:
            # fallback to origin-only route
            path_edges = [origin.getID()]

        vtype = ["hatchback", "sedan", "suv"][idx % 3]
        soc = random.randint(10, 90)
        vehicle_specs.append(
            {
                "vehicle_id": f"ev_{idx}",
                "vtype": vtype,
                "battery_kwh": [30, 45, 60][idx % 3],
                "consumption_wh_per_km": [180, 200, 240][idx % 3],
                "starting_soc_pct": soc,
                "origin_edge": origin.getID(),
                "destination_edge": destination.getID(),
                "route_id": f"route_{idx}",
                "route_edges": path_edges,
            }
        )

    # write routes with staggered depart times
    route_lines = [
        "<routes>",
        '  <vType id="hatchback" accel="2.6" decel="4.5" sigma="0.5" length="4.5" maxSpeed="120"/>',
        '  <vType id="sedan" accel="2.8" decel="4.8" sigma="0.5" length="4.7" maxSpeed="120"/>',
        '  <vType id="suv" accel="2.4" decel="4.2" sigma="0.5" length="4.9" maxSpeed="110"/>',
    ]
    for spec in vehicle_specs:
        route_lines.append(f'  <route id="{spec["route_id"]}" edges="{" ".join(spec["route_edges"])}"/>')

    depart_time = 0
    for spec in vehicle_specs:
        route_lines.append(f'  <vehicle id="{spec["vehicle_id"]}" type="{spec["vtype"]}" route="{spec["route_id"]}" depart="{depart_time}"/>')
        depart_time += 1

    route_lines.append("</routes>")

    OUT_ROUTES.parent.mkdir(parents=True, exist_ok=True)
    OUT_ROUTES.write_text("\n".join(route_lines) + "\n", encoding="utf-8")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(vehicle_specs, indent=2), encoding="utf-8")
    print(f"Wrote {len(vehicle_specs)} EV routes to {OUT_ROUTES}")


def main() -> None:
    build_many_routes(1000)


if __name__ == "__main__":
    main()
