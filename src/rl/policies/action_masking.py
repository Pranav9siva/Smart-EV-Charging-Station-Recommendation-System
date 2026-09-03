"""Action Masking Layer for EV Candidate Station Selection."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch


class ActionMasker:
    """Masks unfeasible candidate stations based on hard safety constraints."""

    def __init__(self, safe_soc_threshold: float = 0.05, max_queue_limit: int = 15) -> None:
        self.safe_soc_threshold = safe_soc_threshold
        self.max_queue_limit = max_queue_limit

    def compute_mask(
        self,
        ev_state: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        candidate_count = len(candidates)
        mask = torch.zeros(candidate_count, dtype=torch.bool)
        masked_count = 0

        ev_soc = float(ev_state.get("battery_pct", 50.0)) / 100.0
        ev_capacity = float(ev_state.get("battery_capacity_kwh", 60.0))

        for idx, cand in enumerate(candidates):
            dist_km = float(cand.get("distance_km", 5.0))
            estimated_energy_kwh = dist_km * 0.20  # 0.20 kWh/km
            remaining_energy_kwh = ev_soc * ev_capacity - estimated_energy_kwh
            remaining_soc_arrival = remaining_energy_kwh / max(1.0, ev_capacity)

            free_ports = int(cand.get("free_ports", 1))
            total_ports = int(cand.get("total_ports", 1))
            queue_len = int(cand.get("queue_len", 0))

            # Mask condition: battery depleted before arrival
            if remaining_soc_arrival < self.safe_soc_threshold:
                mask[idx] = True
                masked_count += 1
                continue

            # Mask condition: zero ports and extreme queue
            if free_ports <= 0 and queue_len >= self.max_queue_limit:
                mask[idx] = True
                masked_count += 1
                continue

        # Ensure at least 1 candidate remains unmasked
        if mask.all():
            mask[0] = False
            masked_count = candidate_count - 1

        valid_count = candidate_count - masked_count
        log_info = {
            "candidate_count": candidate_count,
            "valid_candidate_count": valid_count,
            "masked_candidate_count": masked_count,
        }
        return mask, log_info

    def apply_mask_to_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_logits = logits.clone()
        masked_logits[mask] = -1e9
        return masked_logits
