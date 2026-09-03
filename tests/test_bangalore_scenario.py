import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import sumolib

from src.simulation.bangalore_scenario import generate_bangalore_scenario


def test_generate_bangalore_scenario(tmp_path: Path) -> None:
    result = generate_bangalore_scenario(root=tmp_path, vehicle_count=10)

    assert result["network"].exists()
    assert result["routes"].exists()
    assert result["stations"].exists()
    assert result["config"].exists()

    manifest = json.loads(result["fleet_manifest"].read_text(encoding="utf-8"))
    assert len(manifest) == 10
    for entry in manifest:
        assert 0 < entry["starting_soc_pct"] < 100

    station_lookup = json.loads(result["station_lookup"].read_text(encoding="utf-8"))
    assert len(station_lookup) >= 5
    assert result["route_stats"]["total_requested"] == 10
    assert result["route_stats"]["valid_routes"] == 10
    assert result["route_stats"]["invalid_routes"] == 0


def test_generate_bangalore_scenario_creates_1000_connected_routes(tmp_path: Path) -> None:
    result = generate_bangalore_scenario(root=tmp_path, vehicle_count=1000)

    routes_path = result["routes"]
    net = sumolib.net.readNet(str(result["network"]))
    route_root = ET.parse(routes_path).getroot()
    route_entries = route_root.findall("route")

    assert len(route_entries) == 1000
    assert result["route_stats"]["total_requested"] == 1000
    assert result["route_stats"]["valid_routes"] == 1000
    assert result["route_stats"]["invalid_routes"] == 0
    assert result["route_stats"]["min_route_length"] >= 2
    assert result["route_stats"]["max_route_length"] >= result["route_stats"]["min_route_length"]
    assert result["route_stats"]["average_route_length"] >= 2.0

    routes_by_id = {route.attrib["id"]: route.attrib.get("edges", "").split() for route in route_entries}
    assert len({vehicle.attrib["id"] for vehicle in route_root.findall("vehicle")}) == 1000
    for route_id, edge_ids in routes_by_id.items():
        assert len(edge_ids) >= 2, f"Route {route_id} is too short"
        for prev_id, next_id in zip(edge_ids, edge_ids[1:]):
            prev_edge = net.getEdge(prev_id)
            next_edge = net.getEdge(next_id)
            assert prev_edge.getToNode() == next_edge.getFromNode(), f"Route {route_id} contains disconnected edges {prev_id} -> {next_id}"


def test_bangalore_routes_are_connected() -> None:
    root = Path(__file__).resolve().parents[1]
    network_path = root / "simulations" / "bangalore" / "network.net.xml"
    routes_path = root / "simulations" / "bangalore" / "evs.rou.xml"

    net = sumolib.net.readNet(str(network_path))
    route_root = ET.parse(routes_path).getroot()

    for route in route_root.findall("route"):
        edge_ids = route.attrib.get("edges", "").split()
        assert len(edge_ids) >= 2, f"Route {route.attrib['id']} is too short for realistic city traversal"
        for prev_id, next_id in zip(edge_ids, edge_ids[1:]):
            prev_edge = net.getEdge(prev_id)
            next_edge = net.getEdge(next_id)
            assert prev_edge.getToNode() == next_edge.getFromNode(), (
                f"Route {route.attrib['id']} contains disconnected edges {prev_id} -> {next_id}"
            )
