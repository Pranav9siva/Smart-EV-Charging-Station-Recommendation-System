"""Heterogeneous Graph Builder for EV Charging Station Recommendation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
import torch


class HeteroGraphData:
    """Lightweight Heterogeneous Graph Container."""

    def __init__(self) -> None:
        self.x_dict: Dict[str, torch.Tensor] = {}
        self.edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}

    def to(self, device: torch.device) -> HeteroGraphData:
        data = HeteroGraphData()
        for k, v in self.x_dict.items():
            data.x_dict[k] = v.to(device)
        for k, v in self.edge_index_dict.items():
            data.edge_index_dict[k] = v.to(device)
        return data


class HeteroGraphBuilder:
    """Builds heterogeneous graph representation from live EV & Station simulation state."""

    def __init__(self, candidate_count: int = 8) -> None:
        self.candidate_count = candidate_count

    def build_graph(
        self,
        ev_states: List[Dict[str, Any]],
        candidate_stations_per_ev: List[List[Dict[str, Any]]],
        grid_states: List[Dict[str, Any]] | None = None,
    ) -> HeteroGraphData:
        graph = HeteroGraphData()
        num_evs = len(ev_states)

        if num_evs == 0:
            graph.x_dict["ev"] = torch.zeros((0, 9), dtype=torch.float32)
            graph.x_dict["station"] = torch.zeros((0, 10), dtype=torch.float32)
            graph.x_dict["road"] = torch.zeros((0, 4), dtype=torch.float32)
            graph.x_dict["grid"] = torch.zeros((0, 4), dtype=torch.float32)
            graph.edge_index_dict[("ev", "candidate_at", "station")] = torch.zeros((2, 0), dtype=torch.long)
            graph.edge_index_dict[("ev", "located_on", "road")] = torch.zeros((2, 0), dtype=torch.long)
            graph.edge_index_dict[("station", "connected_to", "grid")] = torch.zeros((2, 0), dtype=torch.long)
            return graph

        # 1. EV Node Features (num_evs, 9)
        ev_feats = []
        for state in ev_states:
            soc = float(state.get("battery_pct", 50.0)) / 100.0
            capacity = float(state.get("battery_capacity_kwh", 60.0)) / 100.0
            range_km = float(state.get("remaining_range_km", 100.0)) / 500.0
            lat = (float(state.get("lat", 12.9716)) - 12.9) * 10.0
            lon = (float(state.get("lon", 77.5946)) - 77.5) * 10.0
            dest_lat = (float(state.get("dest_lat", 12.98)) - 12.9) * 10.0
            dest_lon = (float(state.get("dest_lon", 77.60)) - 77.5) * 10.0
            speed = float(state.get("speed", 10.0)) / 50.0
            step = float(state.get("step", 0)) / 1000.0

            ev_feats.append([soc, capacity, range_km, lat, lon, dest_lat, dest_lon, speed, step])

        graph.x_dict["ev"] = torch.tensor(ev_feats, dtype=torch.float32)

        # 2. Station Node Features & EV -> Station Edges
        station_feats = []
        ev_to_station_edges = []
        station_id_map: Dict[str, int] = {}

        for ev_idx, candidates in enumerate(candidate_stations_per_ev):
            for cand in candidates:
                sid = str(cand.get("station_id", ""))
                if not sid:
                    continue
                if sid not in station_id_map:
                    s_idx = len(station_feats)
                    station_id_map[sid] = s_idx

                    dist = float(cand.get("distance_km", 5.0)) / 50.0
                    ttime = float(cand.get("travel_time_min", 10.0)) / 120.0
                    price = float(cand.get("price_per_kwh", 15.0)) / 50.0
                    free_p = float(cand.get("free_ports", 2)) / 20.0
                    total_p = float(cand.get("total_ports", 4)) / 20.0
                    queue = float(cand.get("queue_len", 0)) / 10.0
                    wait = float(cand.get("avg_wait_min", 0.0)) / 60.0
                    power = float(cand.get("charging_power_kw", cand.get("grid_load_kw", 22.0))) / 150.0
                    gload = float(cand.get("grid_load_kw", 10.0)) / 200.0
                    util = (total_p - free_p) / max(0.05, total_p)

                    station_feats.append([dist, ttime, price, free_p, total_p, queue, wait, power, gload, util])

                s_idx = station_id_map[sid]
                ev_to_station_edges.append([ev_idx, s_idx])

        if not station_feats:
            # Fallback single dummy station node
            station_feats.append([0.1, 0.1, 0.3, 0.1, 0.2, 0.0, 0.0, 0.2, 0.1, 0.5])
            for i in range(num_evs):
                ev_to_station_edges.append([i, 0])

        graph.x_dict["station"] = torch.tensor(station_feats, dtype=torch.float32)

        if ev_to_station_edges:
            edges_t = torch.tensor(ev_to_station_edges, dtype=torch.long).t().contiguous()
        else:
            edges_t = torch.zeros((2, 0), dtype=torch.long)

        graph.edge_index_dict[("ev", "candidate_at", "station")] = edges_t

        # 3. Road / Traffic Nodes (num_evs, 4)
        road_feats = []
        ev_to_road_edges = []
        for i in range(num_evs):
            road_feats.append([0.5, 0.3, 0.2, 0.8])
            ev_to_road_edges.append([i, i])

        graph.x_dict["road"] = torch.tensor(road_feats, dtype=torch.float32)
        graph.edge_index_dict[("ev", "located_on", "road")] = torch.tensor(ev_to_road_edges, dtype=torch.long).t().contiguous()

        # 4. Grid Nodes (num_stations, 4)
        grid_feats = []
        station_to_grid_edges = []
        num_stations = len(station_feats)
        for i in range(num_stations):
            grid_feats.append([0.4, 0.8, 0.5, 0.3])
            station_to_grid_edges.append([i, i])

        graph.x_dict["grid"] = torch.tensor(grid_feats, dtype=torch.float32)
        graph.edge_index_dict[("station", "connected_to", "grid")] = torch.tensor(station_to_grid_edges, dtype=torch.long).t().contiguous()

        return graph
