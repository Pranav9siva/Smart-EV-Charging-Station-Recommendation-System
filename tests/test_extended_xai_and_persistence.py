from __future__ import annotations

from src.xai.shap_explainer import SHAPExplainer
from src.xai.lime_explainer import LIMEExplainer
from src.xai.counterfactual import CounterfactualExplainer
from src.xai.decision_trace import DecisionTrace
from src.persistence.sqlalchemy_repository import SQLAlchemyRepository


def test_shap_explainer_outputs_payload(tmp_path) -> None:
    explainer = SHAPExplainer(tmp_path)
    payload = explainer.explain({"station": "S1", "battery_pct": 20.0, "travel_distance": 2.0, "waiting_time": 1.0, "queue_length": 1, "charging_cost": 5.0})
    assert payload["selected_station"] == "S1"


def test_lime_explainer_outputs_payload(tmp_path) -> None:
    explainer = LIMEExplainer(tmp_path)
    payload = explainer.explain({"station": "S2"})
    assert payload["selected_station"] == "S2"


def test_counterfactual_and_trace() -> None:
    cf = CounterfactualExplainer().build({"station": "S1"})
    trace = DecisionTrace().build({"station": "S1", "confidence": 0.8, "policy_probability": 0.7})
    assert cf["selected_station"] == "S1"
    assert trace["selected_station"] == "S1"


def test_sqlalchemy_repository_initializes(tmp_path) -> None:
    repo = SQLAlchemyRepository(f"sqlite:///{tmp_path / 'test.db'}")
    assert repo.engine is not None
