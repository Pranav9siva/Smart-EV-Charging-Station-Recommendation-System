from __future__ import annotations

import logging
from typing import Any

from src.monitoring.metrics import update_metric
from src.visualization.services.streaming_service import VisualizationService
from src.visualization.services.visualization_builder import VisualizationSnapshotBuilder

logger = logging.getLogger(__name__)


class VisualizationIntegration:
    """Bridge between the controller and the live visualization service."""

    def __init__(self, controller: Any, service: VisualizationService | None = None) -> None:
        self.controller = controller
        # Reuse controller throttle configuration so visualization publication and
        # persistence cadence stay aligned.
        snapshot_record_interval = max(1, int(getattr(controller, "visualization_interval", 1)))
        self.service = service or VisualizationService(output_dir="outputs", snapshot_record_interval=snapshot_record_interval)
        self.builder = VisualizationSnapshotBuilder(controller)

    def publish_step(self, step: int | None = None) -> None:
        snapshot = self.builder.build()
        if step is not None:
            snapshot.setdefault("step", int(step))
            snapshot.setdefault("simulation_time", int(step))
        logger.info("[WS] publish step=%s vehicles=%d stations=%d", step, len(snapshot.get("vehicles", [])), len(snapshot.get("stations", [])))
        self.service.emit_snapshot_sync(snapshot)
        update_metric("websocket_connections", len(getattr(self.service, "_subscribers", [])))
