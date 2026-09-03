from __future__ import annotations

import networkx as nx


class NetworkGraph:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_edge(self, u: str, v: str, travel_time: float, distance: float) -> None:
        self.graph.add_edge(u, v, travel_time=travel_time, distance=distance)

    def shortest_path(self, source: str, target: str, weight: str = "travel_time") -> list[str]:
        return nx.shortest_path(self.graph, source=source, target=target, weight=weight)

    def add_node(self, node_id: str) -> None:
        self.graph.add_node(node_id)

    def get_neighbors(self, node_id: str) -> list[str]:
        return list(self.graph.neighbors(node_id))
