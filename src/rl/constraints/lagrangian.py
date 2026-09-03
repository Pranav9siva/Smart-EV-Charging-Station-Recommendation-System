"""Primal-Dual Lagrangian Constraint System for CMAPPO."""

from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class LagrangianConstraintSystem(nn.Module):
    """Manages learnable dual multipliers lambda_k for 5 explicit constraints."""

    def __init__(
        self,
        num_constraints: int = 5,
        initial_lambda: float = 0.1,
        lr: float = 0.01,
        thresholds: List[float] | None = None,
    ) -> None:
        super().__init__()
        self.num_constraints = num_constraints
        self.lr = lr

        if thresholds is None:
            # Bounds d_1 ... d_5
            thresholds = [0.05, 0.10, 0.10, 0.10, 0.05]

        self.thresholds = torch.tensor(thresholds, dtype=torch.float32)
        # Raw log-multipliers to ensure non-negative lambda = exp(log_lambda) or relu
        self.log_lambdas = nn.Parameter(torch.full((num_constraints,), float(torch.log(torch.tensor(initial_lambda)))))

    @property
    def lambdas(self) -> torch.Tensor:
        return torch.exp(self.log_lambdas).detach()

    def compute_constraint_costs(
        self,
        soc_arrival: float,
        free_ports: int,
        queue_len: int,
        detour_km: float,
        grid_load_kw: float,
    ) -> torch.Tensor:
        c1 = max(0.0, 0.05 - soc_arrival)
        c2 = 1.0 if free_ports <= 0 else 0.0
        c3 = max(0.0, float(queue_len) - 10.0) / 10.0
        c4 = max(0.0, detour_km - 5.0) / 10.0
        c5 = max(0.0, grid_load_kw - 100.0) / 100.0

        return torch.tensor([c1, c2, c3, c4, c5], dtype=torch.float32)

    def update_multipliers(self, avg_costs: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            diff = avg_costs - self.thresholds
            new_lambdas = torch.clamp(self.lambdas + self.lr * diff, min=0.0)
            self.log_lambdas.copy_(torch.log(new_lambdas + 1e-8))

        return {f"lambda_{i+1}": float(l) for i, l in enumerate(self.lambdas)}

    def get_penalty(self, costs: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.lambdas * costs)
