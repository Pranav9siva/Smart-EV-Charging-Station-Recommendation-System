"""Offline Unit Tests for HGAT-CMAPPO Model Components."""

from __future__ import annotations

import tempfile
from pathlib import Path
import torch
import torch.optim as optim
import pytest

from src.rl.hgat.graph_builder import HeteroGraphBuilder
from src.rl.hgat.hgat_encoder import HGATEncoder
from src.rl.mappo.actor_critic import HGAT_CMAPPO_Model, MAPPOActor, MAPPOCritic
from src.rl.policies.action_masking import ActionMasker
from src.rl.constraints.lagrangian import LagrangianConstraintSystem


def test_hetero_graph_builder() -> None:
    builder = HeteroGraphBuilder(candidate_count=4)
    ev_states = [
        {"battery_pct": 30.0, "battery_capacity_kwh": 60.0, "remaining_range_km": 60.0, "lat": 12.97, "lon": 77.59, "dest_lat": 12.98, "dest_lon": 77.60, "speed": 10.0, "step": 5},
        {"battery_pct": 15.0, "battery_capacity_kwh": 50.0, "remaining_range_km": 30.0, "lat": 12.96, "lon": 77.58, "dest_lat": 12.99, "dest_lon": 77.61, "speed": 12.0, "step": 5},
    ]
    candidates = [
        [
            {"station_id": "s1", "distance_km": 2.0, "travel_time_min": 5.0, "price_per_kwh": 12.0, "free_ports": 2, "total_ports": 4, "queue_len": 0, "avg_wait_min": 0.0, "charging_power_kw": 22.0, "grid_load_kw": 10.0},
            {"station_id": "s2", "distance_km": 4.0, "travel_time_min": 8.0, "price_per_kwh": 10.0, "free_ports": 0, "total_ports": 2, "queue_len": 3, "avg_wait_min": 10.0, "charging_power_kw": 50.0, "grid_load_kw": 30.0},
        ],
        [
            {"station_id": "s1", "distance_km": 3.0, "travel_time_min": 6.0, "price_per_kwh": 12.0, "free_ports": 2, "total_ports": 4, "queue_len": 0, "avg_wait_min": 0.0, "charging_power_kw": 22.0, "grid_load_kw": 10.0},
            {"station_id": "s3", "distance_km": 1.0, "travel_time_min": 2.0, "price_per_kwh": 15.0, "free_ports": 1, "total_ports": 2, "queue_len": 0, "avg_wait_min": 0.0, "charging_power_kw": 22.0, "grid_load_kw": 5.0},
        ],
    ]

    graph = builder.build_graph(ev_states, candidates)

    assert "ev" in graph.x_dict
    assert "station" in graph.x_dict
    assert graph.x_dict["ev"].shape == (2, 9)
    assert graph.x_dict["station"].shape[0] == 3  # s1, s2, s3
    assert ("ev", "candidate_at", "station") in graph.edge_index_dict


def test_hgat_encoder_forward() -> None:
    builder = HeteroGraphBuilder(candidate_count=4)
    ev_states = [{"battery_pct": 40.0, "lat": 12.97, "lon": 77.59}]
    candidates = [[{"station_id": "s1", "distance_km": 1.0, "free_ports": 1}]]
    graph = builder.build_graph(ev_states, candidates)

    encoder = HGATEncoder(embedding_dim=64, num_heads=4, num_layers=2)
    h_dict = encoder(graph)

    assert "ev" in h_dict
    assert "station" in h_dict
    assert h_dict["ev"].shape == (1, 64)
    assert h_dict["station"].shape[1] == 64


def test_action_masker() -> None:
    masker = ActionMasker(safe_soc_threshold=0.05, max_queue_limit=5)
    ev_state = {"battery_pct": 5.0, "battery_capacity_kwh": 60.0}  # Very low SOC
    candidates = [
        {"distance_km": 50.0, "free_ports": 0, "queue_len": 10},  # Unreachable
        {"distance_km": 1.0, "free_ports": 2, "queue_len": 0},   # Feasible
    ]

    mask, info = masker.compute_mask(ev_state, candidates)
    assert mask.shape == (2,)
    assert info["candidate_count"] == 2
    assert info["valid_candidate_count"] >= 1


def test_lagrangian_constraint_system() -> None:
    lagrangian = LagrangianConstraintSystem(num_constraints=5, initial_lambda=0.1, lr=0.01)

    costs = lagrangian.compute_constraint_costs(
        soc_arrival=-0.05, # Severe Violation (c1 = 0.10 > 0.05)
        free_ports=0,       # Violation (0 ports)
        queue_len=15,       # Violation (> 10)
        detour_km=8.0,      # Violation (> 5.0)
        grid_load_kw=150.0, # Violation (> 100.0)
    )

    assert costs.shape == (5,)
    assert (costs > 0.0).any()

    penalty = lagrangian.get_penalty(costs)
    assert penalty.item() > 0.0

    mults = lagrangian.update_multipliers(costs)
    assert "lambda_1" in mults
    assert mults["lambda_1"] >= 0.1  # Increased due to violation


def test_hgat_cmappo_model_gradient_flow() -> None:
    model = HGAT_CMAPPO_Model(embedding_dim=64, candidate_count=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    builder = HeteroGraphBuilder(candidate_count=4)
    ev_states = [
        {"battery_pct": 30.0, "battery_capacity_kwh": 60.0, "lat": 12.97, "lon": 77.59},
        {"battery_pct": 50.0, "battery_capacity_kwh": 60.0, "lat": 12.98, "lon": 77.60},
    ]
    candidates = [
        [{"station_id": "s1"}, {"station_id": "s2"}],
        [{"station_id": "s1"}, {"station_id": "s3"}],
    ]

    graph = builder.build_graph(ev_states, candidates)
    cand_indices = [[0, 1], [0, 2]]

    actions, log_probs, values, dist = model(graph, cand_indices)

    loss = -log_probs.mean() + values.mean() ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert actions.shape == (2,)
    assert log_probs.shape == (2,)
    assert values.shape == (1,)
    assert loss.item() != 0.0


def test_model_checkpoint_save_load() -> None:
    model = HGAT_CMAPPO_Model(embedding_dim=64, candidate_count=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "hgat_cmappo.pt"
        torch.save(model.state_dict(), ckpt_path)

        loaded_model = HGAT_CMAPPO_Model(embedding_dim=64, candidate_count=4)
        loaded_model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        assert loaded_model.actor.embedding_dim == 64
