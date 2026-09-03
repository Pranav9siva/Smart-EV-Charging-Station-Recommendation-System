from __future__ import annotations

from pathlib import Path
import sys

# ensure repo root is on sys.path so `src` package imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_env.traci_ev_charging_env import TraciEVChargingEnv


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "simulations" / "bangalore" / "sim.sumocfg"
    # pick up to 2 vehicle ids from the route file if present; here we pass none for a basic smoke test
    env = TraciEVChargingEnv(sumo_config=cfg, tracked_vehicle_ids=[], step_per_action=5)
    try:
        obs, _ = env.reset()
        print("Initial observation keys:", list(obs.keys()))
        for i in range(3):
            obs, reward, done, truncated, info = env.step(action=0)
            print(f"step {i}: reward={reward}, tracked shape={obs['vehicles'].shape}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
