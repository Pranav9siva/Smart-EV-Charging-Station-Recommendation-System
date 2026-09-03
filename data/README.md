# Dataset & Simulation Network Documentation

## Overview

This directory contains real-world OpenStreetMap (OSM) road network extracts, OpenChargeMap EV charging station records, and preprocessed network topologies for the Bengaluru metropolitan simulation domain.

---

## Dataset Inventory

| Dataset Directory | File Name | Description | Source / Provenance |
| :--- | :--- | :--- | :--- |
| `data/raw/` | `bengaluru_charging_stations.csv` | Raw EV charging station POIs in Bengaluru | OpenChargeMap API & OpenStreetMap POIs |
| `data/stations/` | `bengaluru_stations_processed.json` | Preprocessed station objects with port capacities & tariffs | Derived from raw OpenChargeMap records |
| `data/osm/` | `bangalore_subarea.osm` | OSM XML road network extract for Bengaluru | OpenStreetMap export |
| `sumo/network/` | `bengaluru.net.xml` | Compiled SUMO road network file (96 MB) | Compiled via SUMO `netconvert` |

---

## Network & Fleet Attributes

- **Geographic Bounding Box**: $12.83^\circ\text{N} \le \text{Latitude} \le 13.14^\circ\text{N}$, $77.45^\circ\text{E} \le \text{Longitude} \le 77.75^\circ\text{E}$ (Bengaluru Metropolitan Region).
- **EV Fleet Scales Evaluated**: 100 EV, 250 EV, 500 EV, 1,000 EV.
- **Charging Station Infrastructure**:
  - Total Registered Stations: 500+ real charging station locations.
  - Connector Types: CCS2 (Fast DC), Type-2 (AC).
  - Port Capacity per Station: 2 to 12 charging ports per location.
  - Electricity Tariff: Uniform ₹16.00 / kWh tariff base.

---

## Station Processing Script

Raw OpenChargeMap records can be refreshed or parsed using:
```bash
python scripts/fetch_bengaluru_stations.py
```
This queries the OpenChargeMap API within the Bengaluru bounding box and outputs [`data/raw/bengaluru_charging_stations.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/data/raw/bengaluru_charging_stations.csv).
