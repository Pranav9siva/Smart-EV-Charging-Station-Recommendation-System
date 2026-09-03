# Model Checkpoints & Training Artifacts

## Overview

This repository includes pre-trained PyTorch checkpoints for the proposed **Heterogeneous Graph Attention Constraint-Aware Multi-Agent PPO (HGAT-CMAPPO)** policy architecture, as well as baseline PPO variants.

---

## Canonical Model Checkpoint Inventory

| Model Variant | Stage / Seed | Checkpoint Relative Path | Purpose |
| :--- | :--- | :--- | :--- |
| **HGAT-CMAPPO (Proposed)** | Stage 6 (Final) | [`runs/models/hgat_cmappo/stage_06/model.pt`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/models/hgat_cmappo/stage_06/model.pt) | Final validated multi-objective recommendation policy |
| **PPO Baseline** | Final | [`runs/models/ppo/ppo_final.pt`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/models/ppo/ppo_final.pt) | Single-agent baseline evaluation checkpoint |

---

## Multi-Stage Curriculum Training Architecture

The HGAT-CMAPPO model was trained using a 6-stage progressive curriculum:

1. **Stage 01**: Base multi-agent PPO initialization & spatial network embedding.
2. **Stage 02**: Heterogeneous Graph Attention (HGAT) graph neural network encoder integration.
3. **Stage 03**: Lagrangian constraint enforcer activation (queue & grid capacity bounds).
4. **Stage 04**: Demand scaling curriculum (100 $\rightarrow$ 250 EV fleet scale).
5. **Stage 05**: High-demand & station outage perturbation tuning (500 EV scale).
6. **Stage 06 (Final)**: Final joint multi-objective optimization (1,000 EV scale, 5 random seeds: 42, 123, 2024, 31415, 54321).

---

## Loading Pre-Trained Checkpoints in PyTorch

```python
import torch
from src.rl.hgat.hgat_actor_critic import HGATActorCritic

# Initialize policy architecture
policy = HGATActorCritic(
    node_feature_dim=16,
    edge_feature_dim=8,
    hidden_dim=64,
    num_heads=4
)

# Load final Stage 6 checkpoint
checkpoint_path = "runs/models/hgat_cmappo/stage_06/model.pt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    policy.load_state_dict(checkpoint["state_dict"])
else:
    policy.load_state_dict(checkpoint)

policy.eval()
print("[INFO] Successfully loaded HGAT-CMAPPO Stage 6 validated model!")
```
