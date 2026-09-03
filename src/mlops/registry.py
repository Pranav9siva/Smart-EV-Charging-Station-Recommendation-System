from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ModelRegistry:
    """Simple model registry for metadata and versions."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or "outputs/model_registry")
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, model_name: str, version: str, metadata: dict[str, Any]) -> Path:
        payload = {"model_name": model_name, "version": version, "metadata": metadata}
        path = self.root / f"{model_name}_{version}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
