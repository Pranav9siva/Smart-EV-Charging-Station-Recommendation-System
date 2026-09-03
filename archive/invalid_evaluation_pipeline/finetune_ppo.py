from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl_env.gym_ev_charging_env import GymEVChargingEnv

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"
OUTPUT_CHECKPOINT = ROOT / "runs" / "ppo_ckpt" / "ppo_ev_finetuned_50k.zip"
TENSORBOARD_LOG = ROOT / "runs" / "ppo_ev_finetune"
TRACKED_VEHICLE_COUNT = 10
CANDIDATE_COUNT = 8
FINE_TUNE_TIMESTEPS = 50_000


def make_env():
    return GymEVChargingEnv(
        db_path=None,
        tracked_vehicle_count=TRACKED_VEHICLE_COUNT,
        candidate_count=CANDIDATE_COUNT,
    )


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    os.makedirs(OUTPUT_CHECKPOINT.parent, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)

    print("CHECKPOINT_LOADED")
    base_env = make_env()
    env = DummyVecEnv([lambda: base_env])

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = PPO.load(
        str(CHECKPOINT_PATH),
        env=env,
        tensorboard_log=str(TENSORBOARD_LOG),
    )

    observation_match = model.observation_space == base_env.observation_space
    action_match = model.action_space == base_env.action_space

    print(f"OBSERVATION_SPACE_MATCH {'YES' if observation_match else 'NO'}")
    print(f"ACTION_SPACE_MATCH {'YES' if action_match else 'NO'}")
    print(f"FINE_TUNE_TIMESTEPS {FINE_TUNE_TIMESTEPS}")

    if not observation_match or not action_match:
        raise RuntimeError(
            "Loaded checkpoint observation/action spaces do not match the fine-tuning environment"
        )

    model.learn(
        total_timesteps=FINE_TUNE_TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="ppo_ev_finetune",
    )

    model.save(str(OUTPUT_CHECKPOINT))
    print(f"OUTPUT_CHECKPOINT {OUTPUT_CHECKPOINT}")


if __name__ == "__main__":
    main()
