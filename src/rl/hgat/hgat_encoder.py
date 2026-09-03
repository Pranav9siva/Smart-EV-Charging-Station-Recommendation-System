"""Heterogeneous Graph Attention Network (HGAT) Encoder."""

from __future__ import annotations

import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.hgat.graph_builder import HeteroGraphData


class HGATLayer(nn.Module):
    """Single Heterogeneous Graph Attention Layer."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)

        self.attn_drop = nn.Dropout(0.1)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        src_x: torch.Tensor,
        dst_x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0 or dst_x.size(0) == 0:
            return dst_x

        src_idx, dst_idx = edge_index[0], edge_index[1]

        q = self.q_proj(dst_x).view(-1, self.num_heads, self.head_dim)  # (N_dst, H, D)
        k = self.k_proj(src_x).view(-1, self.num_heads, self.head_dim)  # (N_src, H, D)
        v = self.v_proj(src_x).view(-1, self.num_heads, self.head_dim)  # (N_src, H, D)

        q_edge = q[dst_idx]  # (E, H, D)
        k_edge = k[src_idx]  # (E, H, D)
        v_edge = v[src_idx]  # (E, H, D)

        # Scaled Dot-Product Attention over edges
        score = (q_edge * k_edge).sum(dim=-1) / math.sqrt(self.head_dim)  # (E, H)
        alpha = torch.softmax(score, dim=0)
        alpha = self.attn_drop(alpha)

        msg = (v_edge * alpha.unsqueeze(-1)).view(-1, self.out_dim)  # (E, out_dim)

        # Aggregate messages at destination nodes
        out = torch.zeros_like(dst_x)
        out.index_add_(0, dst_idx, msg)

        out = self.out_proj(out)
        return self.norm(dst_x + out)


class HGATEncoder(nn.Module):
    """Heterogeneous Graph Attention Encoder."""

    def __init__(
        self,
        node_feature_dims: Dict[str, int] | None = None,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if node_feature_dims is None:
            node_feature_dims = {"ev": 9, "station": 10, "road": 4, "grid": 4}

        self.embedding_dim = embedding_dim

        # Type-specific linear input encoders
        self.input_encoders = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(dim, embedding_dim),
                nn.ReLU(),
                nn.LayerNorm(embedding_dim),
            )
            for ntype, dim in node_feature_dims.items()
        })

        # Relation Attention Layers
        self.conv1 = HGATLayer(embedding_dim, embedding_dim, num_heads=num_heads)
        self.conv2 = HGATLayer(embedding_dim, embedding_dim, num_heads=num_heads)

    def forward(self, graph: HeteroGraphData) -> Dict[str, torch.Tensor]:
        h_dict: Dict[str, torch.Tensor] = {}

        # 1. Project input features to common embedding dimension
        for ntype, x in graph.x_dict.items():
            if ntype in self.input_encoders:
                h_dict[ntype] = self.input_encoders[ntype](x)
            else:
                h_dict[ntype] = torch.zeros((x.size(0), self.embedding_dim), device=x.device)

        # 2. Relation Message Passing Layer 1
        for rel_key, edge_index in graph.edge_index_dict.items():
            src_type, _, dst_type = rel_key
            if src_type in h_dict and dst_type in h_dict:
                h_dict[dst_type] = self.conv1(h_dict[src_type], h_dict[dst_type], edge_index)

        # 3. Relation Message Passing Layer 2
        for rel_key, edge_index in graph.edge_index_dict.items():
            src_type, _, dst_type = rel_key
            if src_type in h_dict and dst_type in h_dict:
                h_dict[dst_type] = self.conv2(h_dict[src_type], h_dict[dst_type], edge_index)

        return h_dict
