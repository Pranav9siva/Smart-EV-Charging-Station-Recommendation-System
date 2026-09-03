from __future__ import annotations

from typing import Any

from src.rl_env.ev_charging_env import EVChargingEnv
from src.rl_env.ppo_training import train_ppo_policy


def train_and_evaluate_policy(episodes: int = 5, steps_per_episode: int = 10) -> dict[str, Any]:
    env = EVChargingEnv(num_stations=3, num_vehicles=4)
    rewards: list[float] = []

    for _ in range(episodes):
        observation, _ = env.reset()
        episode_reward = 0.0
        for _ in range(steps_per_episode):
            action = env.recommend_action()
            observation, reward, done, _, info = env.step(action)
            episode_reward += float(reward)
            if done:
                break
        rewards.append(episode_reward)

    return {
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "best_reward": max(rewards) if rewards else 0.0,
        "last_selected_station": env.last_selected_station,
    }


def train_and_evaluate_ppo(steps: int = 1000, num_stations: int = 5, num_vehicles: int = 8) -> dict[str, Any]:
    _, metrics = train_ppo_policy(
        steps=steps,
        num_stations=num_stations,
        num_vehicles=num_vehicles,
    )
    return metrics


def train_and_evaluate_ppo_for_test(steps: int = 1000, num_stations: int = 2, num_vehicles: int = 2) -> dict[str, Any]:
    _, metrics = train_ppo_policy(
        steps=steps,
        num_stations=num_stations,
        num_vehicles=num_vehicles,
        ppo_n_steps=min(steps, 64),
        use_test_env=True,
        test_max_episode_steps=min(steps, 64),
    )
    return metrics
