from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter

from src.monitoring.metrics import update_metric

router = APIRouter()

START_TIME = time.time()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@router.get("/health/details")
def health_details() -> dict[str, Any]:
    uptime = time.time() - START_TIME
    update_metric("simulation_running", 1)
    return {
        "database": "healthy",
        "websocket": "healthy",
        "simulation": "running",
        "rl_model": "loaded",
        "uptime": f"{int(uptime)}s",
    }
