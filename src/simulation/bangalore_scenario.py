from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import sumolib

ROOT = Path(__file__).resolve().parents[2]


def _load_real_stations(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                lat = float(row.get("lat", 0.0) or row.get("latitude", 0.0))
                lon = float(row.get("lon", 0.0) or row.get("longitude", 0.0))
            except ValueError:
                continue
            if lat == 0.0 or lon == 0.0:
                continue
            records.append(
                {
                    "station_id": str(row.get("station_id") or row.get("id") or f"real_{len(records)+1}"),
                    "name": str(row.get("name", "Bengaluru Charging Station")),
                    "lat": lat,
                    "lon": lon,
                    "num_ports": max(1, int(row.get("num_ports", row.get("capacity", 1)) or 1)),
                    "power_kw": float(row.get("power_kw", row.get("power", 22.0)) or 22.0),
                }
            )
    return records


def _snap_station_to_lane(net: sumolib.net.Net, lat: float, lon: float) -> tuple[str, float]:
    x, y = net.convertLonLat2XY(lon, lat)
    neighboring_lanes = net.getNeighboringLanes(x, y, 100.0)
    if not neighboring_lanes:
        raise RuntimeError(f"Could not snap station at ({lat},{lon}) to the SUMO network")
    lane_ref = neighboring_lanes[0][0]
    if hasattr(lane_ref, "getID") and hasattr(lane_ref, "getLength"):
        lane = lane_ref
        lane_id = lane.getID()
    else:
        lane_id = str(lane_ref)
        lane = net.getLane(lane_id)
    # choose a start pos that is within lane bounds; prefer 10% into the lane
    lane_len = lane.getLength()
    if lane_len <= 0.5:
        start_pos = max(0.01, lane_len / 2.0)
    else:
        start_pos = min(max(0.01, lane_len * 0.1), max(0.01, lane_len - 0.01))
    return lane_id, start_pos


def _is_connected_edge_sequence(net: sumolib.net.Net, edge_ids: list[str]) -> bool:
    for prev_id, next_id in zip(edge_ids, edge_ids[1:]):
        prev_edge = net.getEdge(prev_id)
        next_edge = net.getEdge(next_id)
        if prev_edge is None or next_edge is None:
            return False
        if prev_edge.getToNode() != next_edge.getFromNode():
            return False
    return True


def _build_route_edge_ids(net: sumolib.net.Net, origin: sumolib.net.edge.Edge, destination: sumolib.net.edge.Edge) -> list[str]:
    try:
        path_edges = net.getShortestPath(origin, destination, vClass="passenger")
        if path_edges:
            if isinstance(path_edges, tuple):
                path_edges = path_edges[0]
            edge_ids = [edge.getID() for edge in path_edges]
            if len(edge_ids) >= 2 and _is_connected_edge_sequence(net, edge_ids):
                return edge_ids
    except Exception:
        pass
    return []


def _find_valid_route_edge_ids(net: sumolib.net.Net, edges: list[sumolib.net.edge.Edge], max_retries: int = 32) -> tuple[list[str], str, str, int]:
    if len(edges) < 2:
        raise RuntimeError("Need at least 2 candidate edges for route generation")

    retries = 0
    origin_edge_id = ""
    destination_edge_id = ""
    for _ in range(max_retries):
        origin = random.choice(edges)
        destination = random.choice(edges)
        while destination == origin:
            destination = random.choice(edges)

        origin_edge_id = origin.getID()
        destination_edge_id = destination.getID()
        route_edge_ids = _build_route_edge_ids(net, origin, destination)
        if len(route_edge_ids) >= 2:
            return route_edge_ids, origin_edge_id, destination_edge_id, retries
        retries += 1

    raise RuntimeError(
        f"Failed to build a valid multi-edge route after {max_retries} attempts for origin={origin_edge_id} destination={destination_edge_id}"
    )


def generate_bangalore_scenario(root: Path | None = None, vehicle_count: int = 10) -> dict[str, Path]:
    root = Path(root or ROOT)
    sim_dir = root / "simulations" / "bangalore"
    data_dir = root / "data"
    station_dir = data_dir / "stations"
    fleet_dir = data_dir / "fleet"
    sim_dir.mkdir(parents=True, exist_ok=True)
    station_dir.mkdir(parents=True, exist_ok=True)
    fleet_dir.mkdir(parents=True, exist_ok=True)

    real_network_path = root / "sumo" / "network" / "bengaluru.net.xml"
    station_csv_path = root / "data" / "raw" / "bengaluru_charging_stations.csv"
    network_path = sim_dir / "network.net.xml"
    nodes_path = sim_dir / "nodes_temp.xml"
    edges_path = sim_dir / "edges_temp.xml"
    routes_path = sim_dir / "evs.rou.xml"
    stations_path = sim_dir / "stations.add.xml"
    config_path = sim_dir / "sim.sumocfg"
    gui_settings_path = sim_dir / "gui_settings.xml"
    station_lookup_path = station_dir / "stations_lookup.json"
    fleet_manifest_path = fleet_dir / "fleet_manifest.json"

    if real_network_path.exists():
        network_path.write_bytes(real_network_path.read_bytes())
        net = sumolib.net.readNet(str(network_path))
    else:
        nodes_path.write_text(
            "<nodes>\n"
            "  <node id=\"J0\" x=\"0.0\" y=\"0.0\"/>\n"
            "  <node id=\"J1\" x=\"3000.0\" y=\"1000.0\"/>\n"
            "  <node id=\"J2\" x=\"6000.0\" y=\"0.0\"/>\n"
            "  <node id=\"J3\" x=\"8500.0\" y=\"1000.0\"/>\n"
            "  <node id=\"J4\" x=\"11000.0\" y=\"0.0\"/>\n"
            "</nodes>\n",
            encoding="utf-8",
        )

        edges_path.write_text(
            "<edges>\n"
            "  <edge id=\"edge0\" from=\"J0\" to=\"J1\" numLanes=\"1\" speed=\"13.89\" priority=\"1\"/>\n"
            "  <edge id=\"edge1\" from=\"J1\" to=\"J2\" numLanes=\"1\" speed=\"13.89\" priority=\"1\"/>\n"
            "  <edge id=\"edge2\" from=\"J2\" to=\"J3\" numLanes=\"1\" speed=\"13.89\" priority=\"1\"/>\n"
            "  <edge id=\"edge3\" from=\"J3\" to=\"J4\" numLanes=\"1\" speed=\"13.89\" priority=\"1\"/>\n"
            "</edges>\n",
            encoding="utf-8",
        )

        from sumolib import checkBinary
        import subprocess

        netconvert = checkBinary("netconvert")
        subprocess.run(
            [
                str(netconvert),
                "--node-files",
                str(nodes_path),
                "--edge-files",
                str(edges_path),
                "-o",
                str(network_path),
                "--no-internal-links",
                "--tls.guess",
            ],
            check=True,
        )
        net = sumolib.net.readNet(str(network_path))

    raw_stations = _load_real_stations(station_csv_path)
    stations: list[dict[str, Any]] = []
    if raw_stations:
        for idx, raw in enumerate(raw_stations[:10], start=1):
            try:
                lane_id, start_pos = _snap_station_to_lane(net, raw["lat"], raw["lon"])
            except RuntimeError:
                continue
            stations.append(
                {
                    "station_id": f"cs_real_{idx}",
                    "name": raw["name"],
                    "num_ports": raw["num_ports"],
                    "power_kw": raw["power_kw"],
                    "lane": lane_id,
                    "start_pos": round(start_pos, 1),
                    "distance_km": 2.0,
                    "travel_time_min": 5.0,
                }
            )

    if not stations and real_network_path.exists():
        valid_lanes = []
        for edge in net.getEdges():
            if edge.allows("passenger"):
                try:
                    valid_lanes.extend(edge.getLanes())
                except AttributeError:
                    pass
        if not valid_lanes:
            for edge in net.getEdges():
                try:
                    valid_lanes.extend(edge.getLanes())
                except AttributeError:
                    pass
        if not valid_lanes:
            raise RuntimeError("Could not enumerate lanes from SUMO network for fallback charging stations")

        for idx, lane in enumerate(valid_lanes[:5], start=1):
            stations.append(
                {
                    "station_id": f"cs_net_{idx}",
                    "name": f"Network Station {idx}",
                    "num_ports": 4,
                    "power_kw": 50.0,
                    "lane": lane.getID(),
                    "start_pos": round((max(0.01, lane.getLength() / 2.0) if lane.getLength() <= 0.5 else min(max(0.01, lane.getLength() * 0.1), max(0.01, lane.getLength() - 0.01))), 1),
                    "distance_km": 2.0,
                    "travel_time_min": 5.0,
                }
            )

    if not stations:
        stations = [
            {
                "station_id": "cs_btm",
                "name": "BTM EV Station",
                "num_ports": 3,
                "power_kw": 50.0,
                "lane": "edge0_0",
                "start_pos": 20,
                "distance_km": 2.0,
                "travel_time_min": 4.0,
            },
            {
                "station_id": "cs_koramangala",
                "name": "Koramangala EV Station",
                "num_ports": 4,
                "power_kw": 50.0,
                "lane": "edge0_0",
                "start_pos": 100,
                "distance_km": 1.5,
                "travel_time_min": 3.0,
            },
            {
                "station_id": "cs_marathahalli",
                "name": "Marathahalli EV Station",
                "num_ports": 2,
                "power_kw": 60.0,
                "lane": "edge1_0",
                "start_pos": 20,
                "distance_km": 3.0,
                "travel_time_min": 6.0,
            },
            {
                "station_id": "cs_whitefield",
                "name": "Whitefield EV Station",
                "num_ports": 2,
                "power_kw": 75.0,
                "lane": "edge2_0",
                "start_pos": 20,
                "distance_km": 4.0,
                "travel_time_min": 7.0,
            },
            {
                "station_id": "cs_indiranagar",
                "name": "Indiranagar EV Station",
                "num_ports": 2,
                "power_kw": 40.0,
                "lane": "edge3_0",
                "start_pos": 20,
                "distance_km": 2.5,
                "travel_time_min": 5.0,
            },
        ]

    station_lookup_path.write_text(json.dumps({entry["station_id"]: entry for entry in stations}, indent=2), encoding="utf-8")

    station_xml_lines = ["<additional>"]
    for station in stations:
        station_xml_lines.append(
            f'  <chargingStation id="{station["station_id"]}" lane="{station["lane"]}" startPos="{station["start_pos"]}" power="{station["power_kw"]}" chargeInTransit="false"/>'
        )
    station_xml_lines.append("</additional>")
    stations_path.write_text("\n".join(station_xml_lines) + "\n", encoding="utf-8")

    passenger_edges = [edge for edge in net.getEdges() if edge.allows("passenger")]
    if len(passenger_edges) < 2:
        passenger_edges = [edge for edge in net.getEdges() if edge.getLaneNumber() > 0]
    if not passenger_edges:
        raise RuntimeError("No valid passenger edges available in the network to generate routes")

    vehicle_specs = []
    random.seed(42)
    soc_values = [18, 22, 24, 31, 36, 42, 55, 63, 71, 82]
    for idx in range(1, int(vehicle_count) + 1):
        vehicle_specs.append(
            {
                "vehicle_id": f"ev_{idx}",
                "vtype": ["hatchback", "sedan", "suv"][idx % 3],
                "battery_kwh": [30, 45, 60][idx % 3],
                "consumption_wh_per_km": [180, 200, 240][idx % 3],
                "starting_soc_pct": soc_values[(idx - 1) % len(soc_values)],
                "origin_edge": "",
                "destination_edge": "",
                "route_id": f"route_{idx}",
                "route_edges": [],
            }
        )

    invalid_specs: list[str] = []
    route_lengths: list[int] = []
    total_retries = 0
    for spec in vehicle_specs:
        try:
            route_edge_ids, origin_edge_id, destination_edge_id, retries = _find_valid_route_edge_ids(net, passenger_edges)
        except Exception:
            invalid_specs.append(str(spec["vehicle_id"]))
            continue

        total_retries += retries
        route_lengths.append(len(route_edge_ids))
        spec["origin_edge"] = origin_edge_id
        spec["destination_edge"] = destination_edge_id
        spec["route_edges"] = route_edge_ids

    if invalid_specs:
        raise RuntimeError(f"Failed to build connected multi-edge routes for {len(invalid_specs)} vehicles: {', '.join(invalid_specs[:10])}")

    route_lines = [
        "<routes>",
        '  <vType id="hatchback" accel="2.6" decel="4.5" sigma="0.5" length="4.5" maxSpeed="120"/>',
        '  <vType id="sedan" accel="2.8" decel="4.8" sigma="0.5" length="4.7" maxSpeed="120"/>',
        '  <vType id="suv" accel="2.4" decel="4.2" sigma="0.5" length="4.9" maxSpeed="110"/>',
    ]
    for spec in vehicle_specs:
        route_lines.append(
            f'  <route id="{spec["route_id"]}" edges="{" ".join(spec["route_edges"])}"/>'
        )
    for depart, spec in enumerate(vehicle_specs):
        route_lines.append(
            f'  <vehicle id="{spec["vehicle_id"]}" type="{spec["vtype"]}" route="{spec["route_id"]}" depart="{depart}"/>'
        )
    route_lines.append("</routes>")
    routes_path.write_text("\n".join(route_lines) + "\n", encoding="utf-8")
    fleet_manifest_path.write_text(json.dumps(vehicle_specs, indent=2), encoding="utf-8")

    config_path.write_text(
        "<configuration>\n"
        "  <input>\n"
        "    <net-file value=\"network.net.xml\"/>\n"
        "    <route-files value=\"evs.rou.xml\"/>\n"
        "    <additional-files value=\"stations.add.xml\"/>\n"
        "  </input>\n"
        "  <time>\n"
        "    <begin value=\"0\"/>\n"
        "    <end value=\"1000\"/>\n"
        "  </time>\n"
        "</configuration>\n",
        encoding="utf-8",
    )

    gui_settings_path.write_text(
        "<viewsettings>\n"
        "  <viewport x=\"5500\" y=\"500\" zoom=\"40\"/>\n"
        "</viewsettings>\n",
        encoding="utf-8",
    )

    route_stats = {
        "total_requested": int(vehicle_count),
        "valid_routes": len(vehicle_specs),
        "invalid_routes": len(invalid_specs),
        "retries": total_retries,
        "min_route_length": min(route_lengths) if route_lengths else 0,
        "max_route_length": max(route_lengths) if route_lengths else 0,
        "average_route_length": round(sum(route_lengths) / len(route_lengths), 3) if route_lengths else 0.0,
    }

    return {
        "network": network_path,
        "routes": routes_path,
        "stations": stations_path,
        "config": config_path,
        "gui_settings": gui_settings_path,
        "station_lookup": station_lookup_path,
        "fleet_manifest": fleet_manifest_path,
        "route_stats": route_stats,
    }
