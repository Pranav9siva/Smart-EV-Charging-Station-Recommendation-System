"""MAPPO Actor-Critic Neural Network Architecture."""

from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.rl.hgat.graph_builder import HeteroGraphData
from src.rl.hgat.hgat_encoder import HGATEncoder
from src.rl.policies.action_masking import ActionMasker


class MAPPOActor(nn.Module):
    """Decentralized Multi-Agent Actor Policy over candidate stations."""

    def __init__(self, embedding_dim: int = 64, candidate_count: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.candidate_count = candidate_count

        # Score matching network combining EV embedding and candidate station embedding
        self.score_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        ev_embeds: torch.Tensor,  # (num_evs, D)
        station_embeds: torch.Tensor,  # (num_stations, D)
        candidate_indices_per_ev: List[List[int]],  # len = num_evs, each has length K
        masks: torch.Tensor | None = None,  # (num_evs, K) boolean mask
    ) -> Tuple[torch.Tensor, Categorical]:
        num_evs = ev_embeds.size(0)
        device = ev_embeds.device

        logits_list = []
        for i in range(num_evs):
            ev_emb = ev_embeds[i]  # (D,)
            c_indices = candidate_indices_per_ev[i]
            K = len(c_indices)

            if K == 0 or station_embeds.size(0) == 0:
                logits_list.append(torch.zeros(self.candidate_count, device=device))
                continue

            c_indices_t = torch.tensor(c_indices, dtype=torch.long, device=device)
            c_embeds = station_embeds[c_indices_t]  # (K, D)

            ev_expanded = ev_emb.unsqueeze(0).expand(K, -1)  # (K, D)
            pair_input = torch.cat([ev_expanded, c_embeds], dim=-1)  # (K, 2D)

            scores = self.score_net(pair_input).squeeze(-1)  # (K,)

            # Pad or truncate to candidate_count
            if K < self.candidate_count:
                pad = torch.full((self.candidate_count - K,), -1e9, device=device)
                scores = torch.cat([scores, pad], dim=0)
            elif K > self.candidate_count:
                scores = scores[: self.candidate_count]

            logits_list.append(scores)

        logits = torch.stack(logits_list, dim=0)  # (num_evs, candidate_count)

        if masks is not None:
            logits[masks] = -1e9

        dist = Categorical(logits=logits)
        return logits, dist


class MAPPOCritic(nn.Module):
    """Centralized Critic Network evaluating global multi-agent state."""

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        # Takes pooled EV embedding + pooled station embedding + pooled road + pooled grid
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        ev_pool = h_dict["ev"].mean(dim=0, keepdim=True) if h_dict["ev"].size(0) > 0 else torch.zeros((1, 64), device=h_dict["ev"].device)
        st_pool = h_dict["station"].mean(dim=0, keepdim=True) if h_dict["station"].size(0) > 0 else torch.zeros((1, 64), device=ev_pool.device)
        rd_pool = h_dict["road"].mean(dim=0, keepdim=True) if h_dict["road"].size(0) > 0 else torch.zeros((1, 64), device=ev_pool.device)
        gr_pool = h_dict["grid"].mean(dim=0, keepdim=True) if h_dict["grid"].size(0) > 0 else torch.zeros((1, 64), device=ev_pool.device)

        global_state = torch.cat([ev_pool, st_pool, rd_pool, gr_pool], dim=-1)  # (1, 4D)
        val = self.net(global_state)  # (1, 1)
        return val.squeeze(-1)


class HGAT_CMAPPO_Model(nn.Module):
    """Full Integrated HGAT-CMAPPO Research Architecture."""

    def __init__(
        self,
        node_feature_dims: Dict[str, int] | None = None,
        embedding_dim: int = 64,
        candidate_count: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = HGATEncoder(node_feature_dims, embedding_dim)
        self.actor = MAPPOActor(embedding_dim, candidate_count)
        self.critic = MAPPOCritic(embedding_dim)
        self.action_masker = ActionMasker()

    def forward(
        self,
        graph: HeteroGraphData,
        candidate_indices_per_ev: List[List[int]],
        masks: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Categorical]:
        h_dict = self.encoder(graph)
        logits, dist = self.actor(h_dict["ev"], h_dict["station"], candidate_indices_per_ev, masks)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.critic(h_dict)
        return actions, log_probs, values, dist
