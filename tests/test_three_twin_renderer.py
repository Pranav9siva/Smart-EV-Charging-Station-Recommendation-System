from __future__ import annotations

import importlib.util
from pathlib import Path


def test_renderer_modules_exist() -> None:
    base = Path('src/visualization/frontend/assets/js/three-twin')
    for filename in ['scene.js', 'camera-controller.js', 'renderer.js']:
        assert (base / filename).exists()


def test_visualization_builder_includes_network_and_stations() -> None:
    from src.visualization.services.visualization_builder import VisualizationSnapshotBuilder

    class DummyController:
        station_manager = type('StationManager', (), {'list_all_stations': lambda self: [], 'get_station_metrics': lambda self, sid: {}, 'get_station_state': lambda self, sid: {}})()
        vehicle_manager = type('VehicleManager', (), {'list_vehicles': lambda self: [], 'list_tracked_vehicles': lambda self: []})()
        recommendation_log = []
        charging_events = []
        simulation_metrics = []
        assignments = {}
        current_step = 0
        last_explanation = None

    builder = VisualizationSnapshotBuilder(DummyController())
    payload = builder.build()
    assert 'network' in payload
    assert 'traffic_lights' in payload


def test_streaming_payload_does_not_mask_top_level_network_with_empty_nested_data() -> None:
    from src.visualization.services.streaming_service import VisualizationService

    service = VisualizationService(output_dir='outputs')
    payload = service._build_payload({
        'network': {'edges': [{'id': 'edge-1', 'points': [[0, 0], [10, 10]]}]},
        'data': {'network': {}, 'vehicles': [], 'stations': []},
    })

    assert payload['data']['network']['edges'][0]['id'] == 'edge-1'
