"""PPO Agent Wrapper reusing existing PPO model baseline for EV Charging Station Recommendation."""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


class PPOPolicy(nn.Module):
    def __init__(self, state_dim: int = 53, action_dim: int = 8):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


class PPOAgent:
    """PPO Agent reusing existing baseline checkpoint or standalone PyTorch PPO."""

    def __init__(
        self,
        state_dim: int = 53,
        action_dim: int = 8,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.sb3_model = None

        self.policy = PPOPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def load_sb3_model(self, model_path: str = "runs/ppo_ckpt/ppo_ev_final.zip") -> bool:
        if PPO is not None and os.path.exists(model_path):
            try:
                self.sb3_model = PPO.load(model_path)
                return True
            except Exception:
                pass
        return False

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[int, float, float]:
        if self.sb3_model is not None:
            try:
                # Try vector box array prediction first
                action, _ = self.sb3_model.predict(state, deterministic=evaluate)
                if isinstance(action, np.ndarray):
                    action = int(action.item()) if action.size == 1 else int(action[0])
                return int(action), 0.0, 0.0
            except Exception:
                try:
                    # Format as Dict observation if SB3 model requires Dict
                    veh_feat = np.zeros((10, 6), dtype=np.float32)
                    veh_feat[0, :min(5, len(state))] = state[:min(5, len(state))]
                    st_feat = state[5:].reshape(-1, 6) if len(state) > 5 else np.zeros((8, 6), dtype=np.float32)
                    dict_obs = {"vehicles": veh_feat, "stations": st_feat}
                    action, _ = self.sb3_model.predict(dict_obs, deterministic=evaluate)
                    if isinstance(action, np.ndarray):
                        action = int(action.item()) if action.size == 1 else int(action[0])
                    return int(action), 0.0, 0.0
                except Exception:
                    pass

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.policy(state_t)
            dist = Categorical(logits=logits)
            if evaluate:
                action = torch.argmax(logits, dim=1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def save(self, filepath: str) -> None:
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        if os.path.exists(filepath):
            try:
                self.policy.load_state_dict(torch.load(filepath, weights_only=False))
                self.policy.eval()
            except Exception:
                pass
