from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl_env.ev_charging_env import EVChargingEnv


def build_env(
    num_stations: int = 5,
    num_vehicles: int = 8,
    prefer_default_db: bool = True,
    max_episode_steps: int | None = None,
) -> gym.Env:
    env = EVChargingEnv(
        num_stations=num_stations,
        num_vehicles=num_vehicles,
        prefer_default_db=prefer_default_db,
        max_episode_steps=max_episode_steps,
    )
    return FlattenObservation(env)


def train_ppo_policy(
    steps: int = 1000,
    num_stations: int = 5,
    num_vehicles: int = 8,
    ppo_n_steps: int | None = None,
    use_test_env: bool = False,
    test_max_episode_steps: int | None = None,
) -> tuple[PPO, dict[str, Any]]:
    env_prefer_default_db = not use_test_env
    env_max_episode_steps = test_max_episode_steps if use_test_env else None

    env = DummyVecEnv(
        [
            lambda: build_env(
                num_stations=num_stations,
                num_vehicles=num_vehicles,
                prefer_default_db=env_prefer_default_db,
                max_episode_steps=env_max_episode_steps,
            )
        ]
    )
    ppo_kwargs: dict[str, Any] = {"verbose": 0}
    if ppo_n_steps is not None:
        ppo_kwargs["n_steps"] = int(ppo_n_steps)
    model = PPO("MlpPolicy", env, **ppo_kwargs)
    model.learn(total_timesteps=steps)
    env.close()

    eval_env = build_env(
        num_stations=num_stations,
        num_vehicles=num_vehicles,
        prefer_default_db=env_prefer_default_db,
        max_episode_steps=env_max_episode_steps,
    )
    episode_rewards: list[float] = []
    for _ in range(5):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = eval_env.step(action)
            episode_reward += float(reward)
        episode_rewards.append(episode_reward)
    eval_env.close()

    metrics = {
        "training_steps": steps,
        "average_eval_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0,
        "max_eval_reward": max(episode_rewards) if episode_rewards else 0.0,
        "min_eval_reward": min(episode_rewards) if episode_rewards else 0.0,
    }
    return model, metrics
