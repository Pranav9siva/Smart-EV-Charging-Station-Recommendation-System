from __future__ import annotations

import json
from pathlib import Path

import sumolib

ROOT = Path(__file__).resolve().parents[1]
NETWORK = ROOT / "simulations" / "bangalore" / "network.net.xml"
STATIONS_JSON = ROOT / "data" / "stations" / "bangalore_stations.json"
OUT_XML = ROOT / "simulations" / "bangalore" / "stations.add.xml"
LOOKUP_JSON = ROOT / "data" / "stations" / "stations_lookup.json"

NETWORK.parent.mkdir(parents=True, exist_ok=True)

if not NETWORK.exists():
    raise FileNotFoundError(f"SUMO network not found at {NETWORK}")

net = sumolib.net.readNet(NETWORK)
stations = json.loads(STATIONS_JSON.read_text(encoding="utf-8")) if STATIONS_JSON.exists() else []

lookup = {}
lines = ["<additional>"]
for station in stations:
    x, y = net.convertLonLat2XY(station["lon"], station["lat"])
    neighboring_lanes = net.getNeighboringLanes(x, y, 20.0)
    lane_id = None
    if neighboring_lanes:
        lane_id = neighboring_lanes[0][0]
    if not lane_id:
        continue
    power_kw = float(station.get("power_kw", 22.0))
    lines.append(
        f'  <chargingStation id="cs_{station["station_id"]}" lane="{lane_id}" startPos="10" endPos="25" power="{power_kw * 1000:.0f}" efficiency="0.95"/>'
    )
    lookup[station["station_id"]] = {
        "lane": lane_id,
        "x": x,
        "y": y,
        "num_ports": int(station.get("num_ports", 1)),
        "power_kw": power_kw,
    }
lines.append("</additional>")
OUT_XML.write_text("\n".join(lines) + "\n", encoding="utf-8")
LOOKUP_JSON.write_text(json.dumps(lookup, indent=2), encoding="utf-8")
print(f"Wrote {len(lookup)} station placements to {OUT_XML}")
