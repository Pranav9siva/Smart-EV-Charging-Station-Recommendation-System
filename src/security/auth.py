from __future__ import annotations

import hashlib
import os
from typing import Any


class SimpleAuth:
    """Minimal authentication layer for API access."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("EV_API_KEY", "dev-key")

    def validate(self, provided_key: str | None) -> bool:
        return bool(provided_key) and hashlib.sha256(provided_key.encode("utf-8")).hexdigest() == hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()
