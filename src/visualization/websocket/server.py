from __future__ import annotations

import json
import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.visualization.services.streaming_service import VisualizationService

router = APIRouter()
service = VisualizationService(output_dir="outputs")
logger = logging.getLogger(__name__)


@router.get("/ws/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "visualization"}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await service.connect(websocket)
    last_timestamp = service.get_last_snapshot().get("timestamp")
    try:
        while True:
            await asyncio.sleep(0.1)
            if not service._load_persisted_snapshot():
                continue
            payload = service.get_last_snapshot()
            timestamp = payload.get("timestamp")
            if timestamp == last_timestamp:
                continue
            last_timestamp = timestamp
            logger.info("[WS] forwarded persisted frame timestamp=%s", timestamp)
            await websocket.send_json(payload)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        await service.disconnect(websocket)
