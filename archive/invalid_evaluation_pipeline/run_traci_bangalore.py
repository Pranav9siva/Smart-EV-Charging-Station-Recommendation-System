from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_ev_fleet import build_routes
from src.simulation.bangalore_scenario import generate_bangalore_scenario
from src.simulation.controller import SimulationController


def prepare_bengaluru_mode(mode: str, tracked: int, depart_window: int, seed: int) -> Path:
    if mode not in {"10", "1000"}:
        raise ValueError("mode must be '10' or '1000'")

    # Always generate station/additional/config artifacts from existing Bengaluru setup.
    paths = generate_bangalore_scenario(root=ROOT, vehicle_count=10)
    if mode == "1000":
        build_routes(vehicle_count=1000, tracked_count=tracked, depart_window=depart_window, seed=seed)
    return paths["config"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["10", "1000"], default="1000", help="Prepare and run 10-EV progression or full 1000-EV Bengaluru flow")
    parser.add_argument("--steps", type=int, default=0, help="Finite step count; 0 runs continuously")
    parser.add_argument("--tracked", type=int, default=10, help="Number of tracked EVs")
    parser.add_argument("--station-count", type=int, default=0, help="Minimum stations in DB; 0 keeps existing DB")
    parser.add_argument("--depart-window", type=int, default=1800, help="Departure spread for 1000-EV route generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for route generation")
    parser.add_argument("--db", default=str(ROOT / "data" / "stations" / "stations.sqlite"))
    parser.add_argument("--model", default=str(ROOT / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"))
    args = parser.parse_args()

    cfg = prepare_bengaluru_mode(args.mode, tracked=args.tracked, depart_window=args.depart_window, seed=args.seed)
    fleet_size = 10 if args.mode == "10" else 1000
    controller = SimulationController(
        sumo_cfg=str(cfg),
        db_path=args.db,
        model_path=args.model,
        fleet_size=fleet_size,
        station_count=args.station_count,
        tracked=args.tracked,
        use_gui=True,
    )
    controller.run(steps=None if args.steps == 0 else args.steps)


if __name__ == "__main__":
    main()
