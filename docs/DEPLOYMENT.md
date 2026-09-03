# Deployment Guide

## Local

```powershell
.venv\Scripts\python.exe -m uvicorn src.visualization.frontend.app:app --host 127.0.0.1 --port 8001
```

Open `/three` for the Digital Twin and `/docs` for the API.

## Docker Compose

```powershell
docker compose up --build
```

The API is exposed on port `8000`; Prometheus uses `9090`; Grafana uses `3000`.

## Kubernetes

Build and publish `ev-digital-twin:latest`, then apply `k8s/deployment.yaml`. Mount a persistent volume at `/app/outputs` for recordings and the research database in production.

## MLflow

Install MLflow separately when required and send `POST /api/research/mlflow/runs`. Without MLflow installed, the adapter returns a local tracking result and does not break the API.
