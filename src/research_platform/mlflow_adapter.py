from __future__ import annotations

from typing import Any


class MLflowTracker:
    """Optional MLflow integration that keeps local experiments usable without MLflow installed."""

    def __init__(self, tracking_uri: str | None = None, experiment_name: str = "ev-digital-twin") -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.available = False
        self._mlflow: Any = None
        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
            self.available = True
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
        except ImportError:
            pass

    def log_run(self, parameters: dict[str, Any], metrics: dict[str, float], tags: dict[str, str] | None = None) -> dict[str, Any]:
        payload = {"parameters": parameters, "metrics": metrics, "tags": tags or {}, "backend": "mlflow" if self.available else "local"}
        if self.available and self._mlflow:
            with self._mlflow.start_run() as run:
                self._mlflow.log_params(parameters)
                self._mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
                self._mlflow.set_tags(tags or {})
                payload["run_id"] = run.info.run_id
        return payload
