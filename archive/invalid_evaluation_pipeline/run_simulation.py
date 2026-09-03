"""Start the real-time simulation loop using SimulationController.

Usage:
    python scripts/run_simulation.py --sumo-cfg simulations/example/sim.sumocfg --db data/stations/stations.sqlite --model runs/ppo_ckpt/ppo_ev_final.zip
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT.resolve())
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from src.simulation.controller import SimulationController


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sumo-cfg", default=str(ROOT / "simulations" / "bangalore" / "sim.sumocfg"))
    p.add_argument("--db", default=str(ROOT / "data" / "stations" / "stations.sqlite"))
    p.add_argument("--model", default=str(ROOT / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"))
    p.add_argument("--steps", type=int, default=0, help="Finite step count; 0 runs continuously until stopped")
    p.add_argument("--fleet-size", type=int, default=1000, help="VehicleManager fleet size")
    p.add_argument("--tracked", type=int, default=10, help="Number of tracked EVs in dashboard/live view")
    p.add_argument("--station-count", type=int, default=0, help="Minimum stations required in DB; 0 means use existing DB as-is")
    p.add_argument("--gui", action="store_true", help="Run SUMO-GUI under TraCI control")
    p.add_argument("--dashboard-state-interval", type=int, default=5, help="Refresh dashboard state every N simulation steps")
    p.add_argument("--dashboard-report-interval", type=int, default=10, help="Write dashboard report artifacts every N dashboard updates")
    p.add_argument("--visualization-interval", type=int, default=5, help="Publish visualization snapshots every N simulation steps")
    p.add_argument("--metrics-interval", type=int, default=5, help="Collect metrics every N simulation steps")
    p.add_argument("--prepare-bangalore", choices=["none", "10", "1000"], default="none", help="Prepare Bengaluru scenario for progression testing")
    p.add_argument("--depart-window", type=int, default=1800, help="When preparing 1000 EVs, spread departures across this many steps")
    p.add_argument("--seed", type=int, default=42, help="Random seed for generated Bengaluru fleet")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    sumo_cfg_path = Path(args.sumo_cfg)
    if args.prepare_bangalore != "none":
        from src.simulation.bangalore_scenario import generate_bangalore_scenario

        prepared = generate_bangalore_scenario(root=ROOT, vehicle_count=10)
        sumo_cfg_path = prepared["config"]

        if args.prepare_bangalore == "10" and args.fleet_size == 1000:
            args.fleet_size = 10
        if args.prepare_bangalore == "1000":
            from scripts.generate_ev_fleet import build_routes

            if args.fleet_size < 1000:
                args.fleet_size = 1000
            build_routes(vehicle_count=1000, tracked_count=args.tracked, depart_window=args.depart_window, seed=args.seed)

    ctrl = SimulationController(
        sumo_cfg=str(sumo_cfg_path),
        db_path=args.db,
        model_path=args.model,
        fleet_size=args.fleet_size,
        station_count=args.station_count,
        tracked=args.tracked,
        use_gui=args.gui,
        dashboard_state_interval=args.dashboard_state_interval,
        dashboard_report_interval=args.dashboard_report_interval,
        visualization_interval=args.visualization_interval,
        metrics_interval=args.metrics_interval,
    )
    ctrl.run(steps=None if args.steps == 0 else args.steps)


if __name__ == '__main__':
    main()
