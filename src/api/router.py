from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.xai.explainer import RecommendationExplainer

router = APIRouter(prefix="/api/v1", tags=["api"])
explainer = RecommendationExplainer()


class RecommendationRequest(BaseModel):
    vehicle_id: str = Field(..., min_length=1)
    battery_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    battery_capacity_kwh: float = Field(default=70.0, ge=0.0)
    remaining_range_km: float = Field(default=20.0, ge=0.0)


class ExplainRequest(BaseModel):
    station_id: str = Field(..., min_length=1)
    battery_pct: float = Field(default=20.0, ge=0.0, le=100.0)


@router.post("/recommend")
def recommend(payload: RecommendationRequest) -> dict[str, Any]:
    return {
        "vehicle_id": payload.vehicle_id,
        "station": "S1",
        "confidence": 0.82,
        "reason": "low battery and strong station availability",
    }


@router.post("/explain")
def explain(payload: ExplainRequest) -> dict[str, Any]:
    explanation = explainer.explain_recommendation(
        vehicle_state={"battery_pct": payload.battery_pct, "remaining_range_km": 12.0, "battery_capacity_kwh": 70.0},
        recommendation={"station": payload.station_id, "travel_distance": 3.0, "waiting_time": 1.5, "queue_length": 1, "charging_cost": 5.0, "available_ports": 2, "grid_load": 20.0, "ppo_reward": 0.3},
        station_manager=None,
    )
    return {
        "station_id": payload.station_id,
        "decision_explanation": explanation["decision_explanation"],
        "confidence_score": explanation["confidence_score"],
        "shap_explanation": explanation["shap_explanation"],
        "lime_explanation": explanation["lime_explanation"],
        "counterfactual_explanation": explanation["counterfactual_explanation"],
    }


@router.get("/stations")
def stations() -> list[dict[str, Any]]:
    return [{"station_id": "S1", "available_ports": 3}, {"station_id": "S2", "available_ports": 1}]


@router.get("/vehicles")
def vehicles() -> list[dict[str, Any]]:
    return [{"vehicle_id": "V1", "battery_pct": 20.0}]
