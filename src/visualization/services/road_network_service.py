from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


class RoadNetworkService:
    """Expose a simplified SUMO road network for the Three.js frontend."""

    def __init__(self, network_path: str | Path | None = None) -> None:
        self.network_path = Path(network_path or "simulations/bangalore/network.net.xml")
        # Cache static network payload and refresh only when file mtime changes.
        self._cached_payload: dict[str, Any] | None = None
        self._cached_mtime_ns: int | None = None

    def build_payload(self) -> dict[str, Any]:
        path = self._resolve_path()
        if path.exists():
            try:
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                mtime_ns = None
            if self._cached_payload is not None and self._cached_mtime_ns is not None and mtime_ns == self._cached_mtime_ns:
                return self._cached_payload
        if not path.exists():
            return {"edges": [], "bounds": {"minX": 0, "maxX": 100, "minY": 0, "maxY": 100}}

        tree = ET.parse(path)
        root = tree.getroot()
        geo_bounds: dict[str, float] = {}
        location = root.find("location")
        if location is not None:
            try:
                orig_min_x, orig_min_y, orig_max_x, orig_max_y = [float(value) for value in location.get("origBoundary", "").split(",")]
                conv_min_x, conv_min_y, conv_max_x, conv_max_y = [float(value) for value in location.get("convBoundary", "").split(",")]
                geo_bounds = {
                    "minLon": orig_min_x,
                    "minLat": orig_min_y,
                    "maxLon": orig_max_x,
                    "maxLat": orig_max_y,
                    "minX": conv_min_x,
                    "minY": conv_min_y,
                    "maxX": conv_max_x,
                    "maxY": conv_max_y,
                }
            except (TypeError, ValueError):
                geo_bounds = {}
        nodes: dict[str, tuple[float, float]] = {}
        for junction in root.findall("junction"):
            node_id = junction.get("id")
            if not node_id:
                continue
            try:
                x = float(junction.get("x", 0.0))
                y = float(junction.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            nodes[node_id] = (x, y)

        edges: list[dict[str, Any]] = []
        for edge in root.findall("edge"):
            edge_id = edge.get("id") or ""
            if not edge_id or edge_id.startswith(":"):
                continue
            from_node = edge.get("from")
            to_node = edge.get("to")
            raw_shape = edge.get("shape", "")
            points: list[tuple[float, float]] = []
            if raw_shape:
                pieces = [segment for segment in raw_shape.split() if segment]
                for segment in pieces:
                    try:
                        x, y = segment.split(",")
                        points.append((float(x), float(y)))
                    except ValueError:
                        continue
            else:
                start = nodes.get(from_node)
                end = nodes.get(to_node)
                if start and end:
                    points = [start, end]
            if len(points) < 2:
                continue
            edges.append({"id": edge_id, "points": points})
            if len(edges) >= 2500:
                break

        bounds = self._compute_bounds(edges)
        payload = {"edges": edges, "bounds": bounds, "geo_bounds": geo_bounds}
        try:
            self._cached_mtime_ns = path.stat().st_mtime_ns
        except OSError:
            self._cached_mtime_ns = None
        self._cached_payload = payload
        return payload

    def _resolve_path(self) -> Path:
        candidates = [self.network_path]
        root = Path(__file__).resolve().parents[3]
        candidates.append(root / self.network_path)
        candidates.append(root / "simulations" / "bangalore" / "network.net.xml")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return self.network_path

    def _compute_bounds(self, edges: list[dict[str, Any]]) -> dict[str, float]:
        all_points = [point for edge in edges for point in edge.get("points", [])]
        if not all_points:
            return {"minX": 0.0, "maxX": 100.0, "minY": 0.0, "maxY": 100.0}
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return {
            "minX": min(xs),
            "maxX": max(xs),
            "minY": min(ys),
            "maxY": max(ys),
            "width": width or 1.0,
            "height": height or 1.0,
        }
