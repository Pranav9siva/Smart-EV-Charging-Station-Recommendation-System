from __future__ import annotations

from src.rl_env.gym_ev_charging_env import GymEVChargingEnv


def test_gym_env_basic_api() -> None:
    env = GymEVChargingEnv(db_path=None, station_manager=None, tracked_vehicle_count=5, candidate_count=4)
    obs, _ = env.reset()
    assert "vehicles" in obs and "stations" in obs
    assert obs["vehicles"].shape == (5, 6)
    assert obs["stations"].shape == (4, 6)

    # sample an action and step
    action = env.action_space.sample()
    obs2, reward, done, truncated, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(info, dict)
    env.close()
