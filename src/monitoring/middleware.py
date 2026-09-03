from __future__ import annotations

import time
from typing import Callable

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.monitoring.metrics import increment_counter, observe_histogram, update_metric


class MonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> object:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        increment_counter("api_requests_total")
        observe_histogram("api_response_time", duration_ms)
        update_metric("websocket_connections", 0)
        return response


def register_monitoring_middleware(app: FastAPI) -> None:
    app.add_middleware(MonitoringMiddleware)
