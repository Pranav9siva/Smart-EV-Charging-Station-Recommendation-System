from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation.controller import SimulationController


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Smart EV charging simulation")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fleet-size", type=int, default=10)
    parser.add_argument("--tracked", type=int, default=10)
    parser.add_argument("--sumo-cfg", default="simulations/bangalore/sim.sumocfg")
    parser.add_argument("--db", default="data/stations/stations.sqlite")
    parser.add_argument("--model", default="runs/ppo_ckpt/ppo_ev_final.zip")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args(argv)

    controller = SimulationController(
        sumo_cfg=str(Path(args.sumo_cfg)),
        db_path=str(Path(args.db)),
        model_path=str(Path(args.model)),
        fleet_size=args.fleet_size,
        tracked=args.tracked,
        use_gui=args.gui,
    )
    try:
        controller.run(steps=args.steps)
    finally:
        controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())