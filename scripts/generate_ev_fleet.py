from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import sumolib
from sumolib import checkBinary

from src.ev_management.vehicle_manager import VehicleManager

ROOT = Path(__file__).resolve().parents[1]
NETWORK = ROOT / "simulations" / "bangalore" / "network.net.xml"
OUT_ROUTES = ROOT / "simulations" / "bangalore" / "evs.rou.xml"
MANIFEST = ROOT / "data" / "fleet" / "fleet_manifest.json"


def _normalize_route_edges(route_data: Any) -> list[str]:
    if not route_data:
        return []
    if isinstance(route_data, tuple):
        route_data = route_data[0]
    edge_ids: list[str] = []
    for edge in route_data:
        edge_ids.append(edge if isinstance(edge, str) else edge.getID())
    return edge_ids


def _is_connected_edge_sequence(net: sumolib.net.Net, edge_ids: list[str]) -> bool:
    if len(edge_ids) < 2:
        return False
    for prev_id, next_id in zip(edge_ids, edge_ids[1:]):
        prev_edge = net.getEdge(prev_id)
        if prev_edge is None:
            return False
        outgoing = prev_edge.getOutgoing()
        if not outgoing:
            return False
        connected_ids = {edge.getID() for edge in outgoing.keys()}
        if next_id not in connected_ids:
            return False
    return True


def _build_edge_adjacency(net: sumolib.net.Net, edges: list[Any]) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        outgoing = edge.getOutgoing() or {}
        adjacency[edge.getID()] = [next_edge.getID() for next_edge in outgoing.keys() if next_edge is not None]
    return adjacency


def _find_valid_route_edge_ids(
    net: sumolib.net.Net,
    edges: list[Any],
    adjacency: dict[str, list[str]],
    *,
    max_retries: int = 32,
) -> tuple[list[str], str, str, int]:
    if len(edges) < 2:
        raise RuntimeError("Need at least 2 candidate edges for route generation")

    retries = 0
    origin_edge_id = ""
    destination_edge_id = ""
    for _ in range(max_retries):
        origin = random.choice(edges)
        origin_edge_id = origin.getID()

        reachable: set[str] = set()
        pending = list(adjacency.get(origin_edge_id, []))
        while pending:
            next_edge_id = pending.pop()
            if next_edge_id in reachable or next_edge_id == origin_edge_id:
                continue
            reachable.add(next_edge_id)
            pending.extend(adjacency.get(next_edge_id, []))

        reachable = {edge_id for edge_id in reachable if edge_id != origin_edge_id}
        if not reachable:
            retries += 1
            continue

        destination_edge_id = random.choice(sorted(reachable))
        destination = net.getEdge(destination_edge_id)
        if destination is None:
            retries += 1
            continue

        try:
            route_data = net.getShortestPath(origin, destination, vClass="passenger")
            route_edge_ids = _normalize_route_edges(route_data)
        except Exception:
            route_edge_ids = []

        if len(route_edge_ids) >= 2 and _is_connected_edge_sequence(net, route_edge_ids):
            return route_edge_ids, origin_edge_id, destination_edge_id, retries

        retries += 1

    raise RuntimeError(
        f"Failed to build a valid multi-edge route after {max_retries} attempts for origin={origin_edge_id} destination={destination_edge_id}"
    )


def _is_sumo_route_connected(net: sumolib.net.Net, edge_ids: list[str]) -> bool:
    if len(edge_ids) <= 1:
        return True
    for prev_id, next_id in zip(edge_ids, edge_ids[1:]):
        prev_edge = net.getEdge(prev_id)
        if prev_edge is None:
            return False
        outgoing = prev_edge.getOutgoing()
        if not outgoing:
            return False
        connected_ids = {edge.getID() for edge in outgoing.keys()}
        if next_id not in connected_ids:
            return False
    return True


def _build_departure_schedule(vehicle_count: int, depart_window: int) -> list[int]:
    if vehicle_count <= 0:
        return []
    safe_depart_window = max(1, int(depart_window))
    denom = max(1, int(vehicle_count) - 1)
    return [int((index * safe_depart_window) / denom) for index in range(int(vehicle_count))]


def _compute_departure_stats(departure_steps: list[int], bucket_size: int = 10) -> dict[str, Any]:
    if not departure_steps:
        return {
            "first_departure": 0,
            "last_departure": 0,
            "average_departure_time": 0.0,
            "vehicles_per_time_bucket": {},
            "max_vehicles_departing_in_one_step": 0,
        }

    safe_bucket_size = max(1, int(bucket_size))
    departures_by_step = Counter(departure_steps)
    departures_by_bucket = Counter((step // safe_bucket_size) * safe_bucket_size for step in departure_steps)
    return {
        "first_departure": int(min(departure_steps)),
        "last_departure": int(max(departure_steps)),
        "average_departure_time": round(sum(departure_steps) / len(departure_steps), 3),
        "vehicles_per_time_bucket": {str(key): int(value) for key, value in sorted(departures_by_bucket.items())},
        "max_vehicles_departing_in_one_step": int(max(departures_by_step.values())),
    }


def build_routes(vehicle_count: int = 1000, tracked_count: int = 10, depart_window: int = 120, seed: int = 42) -> dict[str, Any]:
    if not NETWORK.exists():
        raise FileNotFoundError(f"SUMO network not found at {NETWORK}")

    net = sumolib.net.readNet(NETWORK)
    edges = [edge for edge in net.getEdges() if edge.allows("passenger")]
    if len(edges) < 10:
        raise RuntimeError("Need at least 10 candidate edges")

    manager = VehicleManager()
    vehicle_specs = []
    random.seed(seed)
    for idx in range(1, int(vehicle_count) + 1):
        vehicle_specs.append(
            {
                "vehicle_id": f"ev_{idx}",
                "vtype": ["hatchback", "sedan", "suv"][idx % 3],
                "battery_kwh": [30, 45, 60][idx % 3],
                "consumption_wh_per_km": [180, 200, 240][idx % 3],
                "starting_soc_pct": random.randint(5, 95),
                "origin_edge": "",
                "destination_edge": "",
                "route_id": f"route_{idx}",
                "route_edges": [],
            }
        )

    adjacency = _build_edge_adjacency(net, edges)
    invalid_specs: list[str] = []
    total_retries = 0
    route_lengths: list[int] = []
    for spec in vehicle_specs:
        try:
            route_edge_ids, origin_edge_id, destination_edge_id, retries = _find_valid_route_edge_ids(net, edges, adjacency)
        except Exception:
            invalid_specs.append(str(spec["vehicle_id"]))
            continue

        total_retries += retries
        route_lengths.append(len(route_edge_ids))
        spec["origin_edge"] = origin_edge_id
        spec["destination_edge"] = destination_edge_id
        spec["route_edges"] = route_edge_ids

    if invalid_specs:
        raise RuntimeError(
            f"Failed to build connected multi-edge routes for {len(invalid_specs)} vehicles: {', '.join(invalid_specs[:10])}"
        )

    route_lines = [
        "<routes>",
        '  <vType id="hatchback" accel="2.6" decel="4.5" sigma="0.5" length="4.5" maxSpeed="120"/>',
        '  <vType id="sedan" accel="2.8" decel="4.8" sigma="0.5" length="4.7" maxSpeed="120"/>',
        '  <vType id="suv" accel="2.4" decel="4.2" sigma="0.5" length="4.9" maxSpeed="110"/>',
    ]
    for spec in vehicle_specs:
        route_lines.append(f'  <route id="{spec["route_id"]}" edges="{" ".join(spec["route_edges"])}"/>')
    departure_steps = _build_departure_schedule(len(vehicle_specs), depart_window)
    departure_stats = _compute_departure_stats(departure_steps, bucket_size=10)
    for index, spec in enumerate(vehicle_specs):
        depart_step = departure_steps[index]
        route_lines.append(f'  <!-- {spec["vehicle_id"]}: starting SoC {spec["starting_soc_pct"]}% -->')
        route_lines.append(f'  <vehicle id="{spec["vehicle_id"]}" type="{spec["vtype"]}" route="{spec["route_id"]}" depart="{depart_step}"/>')
    route_lines.append("</routes>")
    lines = route_lines

    OUT_ROUTES.parent.mkdir(parents=True, exist_ok=True)
    OUT_ROUTES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manager.generate_fleet(
        count=int(vehicle_count),
        detail_count=max(1, min(int(tracked_count), int(vehicle_count))),
        sources=[f"source_{i}" for i in range(1, 51)],
        destinations=[f"destination_{i}" for i in range(1, 51)],
    )
    manager.save_manifest(MANIFEST)
    route_stats = {
        "total_requested": int(vehicle_count),
        "valid_routes": len(vehicle_specs),
        "invalid_routes": len(invalid_specs),
        "retries": total_retries,
        "min_route_length": min(route_lengths) if route_lengths else 0,
        "max_route_length": max(route_lengths) if route_lengths else 0,
        "average_route_length": round(sum(route_lengths) / len(route_lengths), 3) if route_lengths else 0.0,
        **departure_stats,
    }
    print(f"Wrote {len(vehicle_specs)} EV routes to {OUT_ROUTES}")
    print(f"Saved {vehicle_count}-vehicle fleet manifest with {max(1, min(int(tracked_count), int(vehicle_count)))} tracked vehicles to {MANIFEST}")
    print(json.dumps(route_stats, indent=2))
    return route_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle-count", type=int, default=1000, help="Number of EVs to include in the Bengaluru routes file")
    parser.add_argument("--tracked", type=int, default=10, help="Number of tracked EVs in the fleet manifest")
    parser.add_argument("--depart-window", type=int, default=120, help="Spread departures over this many simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible route generation")
    args = parser.parse_args()
    build_routes(vehicle_count=args.vehicle_count, tracked_count=args.tracked, depart_window=args.depart_window, seed=args.seed)


if __name__ == "__main__":
    main()
