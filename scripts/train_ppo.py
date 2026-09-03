"""Train PPO on the Gym EV charging environment.

Saves periodic checkpoints and writes TensorBoard logs.
Run:
    python scripts/train_ppo.py --timesteps 2000
"""
from __future__ import annotations

import argparse
import os
import time
from functools import partial

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
except Exception as e:  # pragma: no cover - environment may not have sb3 installed
    raise

from src.rl_env.gym_ev_charging_env import GymEVChargingEnv


def make_env_factory(db_path: str | None, tracked: int, candidates: int):
    def _init():
        env = GymEVChargingEnv(db_path=db_path, tracked_vehicle_count=tracked, candidate_count=candidates)
        return env

    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2000)
    p.add_argument("--logdir", type=str, default="runs/ppo_ev")
    p.add_argument("--checkpoint-freq", type=int, default=500)
    p.add_argument("--tracked", type=int, default=10)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--db-path", type=str, default=None)
    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    ckpt_dir = os.path.join(args.logdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # make vectorized env
    env_fn = make_env_factory(args.db_path, args.tracked, args.candidates)
    env = DummyVecEnv([env_fn])

    # policy: MultiInputPolicy for Dict obs
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.logdir,
    )

    # checkpoint callback
    ckpt_cb = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=ckpt_dir, name_prefix="ppo_ev")

    start = time.time()
    print(f"Starting training for {args.timesteps} timesteps (this will run briefly for validation)...")
    model.learn(total_timesteps=args.timesteps, callback=ckpt_cb, tb_log_name="ppo_ev")
    duration = time.time() - start

    final_path = os.path.join(args.logdir, "ppo_ev_final.zip")
    model.save(final_path)
    print(f"Training finished in {duration:.1f}s, saved final model to: {final_path}")


if __name__ == "__main__":
    main()
