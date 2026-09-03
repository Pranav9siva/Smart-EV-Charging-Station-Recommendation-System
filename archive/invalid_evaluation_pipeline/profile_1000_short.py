from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import traci
from sumolib import checkBinary
from sumolib.miscutils import getFreeSocketPort

from scripts.generate_ev_fleet import build_routes
from src.simulation.controller import SimulationController

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "simulations" / "bangalore" / "sim.sumocfg"
DB = ROOT / "data" / "stations" / "stations.sqlite"
MODEL = ROOT / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"
OUTPUT = ROOT / "outputs" / "short_1000_profile.json"
MINIMAL_ERROR_LOG = ROOT / "outputs" / "short_1000_minimal_sumo_errors.log"


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _count_routes(route_file: Path) -> int:
    text = route_file.read_text(encoding="utf-8")
    return sum(1 for line in text.splitlines() if '<vehicle id="ev_' in line)


def run_minimal_10_step(*, use_gui: bool, steps: int) -> dict[str, Any]:
    sumo_binary = checkBinary("sumo-gui" if use_gui else "sumo")
    port = getFreeSocketPort()
    if port is None:
        raise RuntimeError("Failed to allocate free port for minimal run")

    cmd = [
        sumo_binary,
        "-c",
        str(CFG),
        "--error-log",
        str(MINIMAL_ERROR_LOG),
        "--no-step-log",
        "true",
        "--verbose",
        "false",
        "--time-to-teleport",
        "-1",
    ]

    started_ms = _now_ms()
    traci.start(cmd, port=int(port), numRetries=120, verbose=False)
    start_duration_ms = _now_ms() - started_ms

    departed_seen: set[str] = set()
    moved_seen: set[str] = set()
    seen_vehicle_ids: set[str] = set()
    step_rows: list[dict[str, Any]] = []

    try:
        for step in range(steps):
            step_start = _now_ms()
            traci.simulationStep()
            step_duration_ms = _now_ms() - step_start

            active_ids = traci.vehicle.getIDList()
            loaded_now = traci.simulation.getLoadedIDList()
            departed_now = traci.simulation.getDepartedIDList()
            pending_ids = traci.simulation.getPendingVehicles()

            departed_seen.update(departed_now)
            seen_vehicle_ids.update(active_ids)
            seen_vehicle_ids.update(loaded_now)
            seen_vehicle_ids.update(departed_now)
            for vid in active_ids:
                try:
                    if traci.vehicle.getSpeed(vid) > 0.1:
                        moved_seen.add(vid)
                except traci.TraCIException:
                    continue

            step_rows.append(
                {
                    "step": step,
                    "simulation_time": traci.simulation.getTime(),
                    "active_count": len(active_ids),
                    "loaded_this_step": len(loaded_now),
                    "departed_this_step": len(departed_now),
                    "pending_count": len(pending_ids),
                    "moving_count": len(moved_seen),
                    "step_ms": round(step_duration_ms, 3),
                }
            )

        total_known_after_steps = len(seen_vehicle_ids) + len(traci.simulation.getPendingVehicles())
        route_count = _count_routes(ROOT / "simulations" / "bangalore" / "evs.rou.xml")
        error_log = ""
        if MINIMAL_ERROR_LOG.exists():
            error_log = MINIMAL_ERROR_LOG.read_text(encoding="utf-8", errors="ignore")

        return {
            "sumo_binary": str(sumo_binary),
            "start_duration_ms": round(start_duration_ms, 3),
            "steps": int(steps),
            "route_vehicle_count": route_count,
            "seen_vehicle_ids_count": len(seen_vehicle_ids),
            "known_vehicle_count_after_steps": total_known_after_steps,
            "departed_unique_count": len(departed_seen),
            "moved_unique_count": len(moved_seen),
            "step_rows": step_rows,
            "route_errors_detected": ("no valid route" in error_log.lower()) or ("error:" in error_log.lower()),
            "error_log_tail": "\n".join(error_log.splitlines()[-40:]),
        }
    finally:
        try:
            traci.close()
        except Exception:
            pass


def run_controller_stage_profile(*, use_gui: bool, steps: int) -> dict[str, Any]:
    ctrl = SimulationController(
        sumo_cfg=str(CFG),
        db_path=str(DB),
        model_path=str(MODEL),
        fleet_size=1000,
        station_count=0,
        tracked=10,
        charge_threshold_pct=-1.0,
        use_gui=use_gui,
    )

    stage: dict[str, Any] = {
        "ctrl_start_ms": None,
        "step_profiles": [],
    }

    start_t0 = _now_ms()
    ctrl.start()
    stage["ctrl_start_ms"] = round(_now_ms() - start_t0, 3)

    try:
        for step in range(steps):
            row: dict[str, Any] = {"step": step}

            t0 = _now_ms()
            traci.simulationStep()
            row["simulation_step_ms"] = round(_now_ms() - t0, 3)

            t0 = _now_ms()
            ctrl._update_vehicles(step)
            row["update_vehicles_ms"] = round(_now_ms() - t0, 3)

            t0 = _now_ms()
            ctrl._process_charging(step)
            row["process_charging_ms"] = round(_now_ms() - t0, 3)

            t0 = _now_ms()
            ctrl._refresh_dashboard()
            row["refresh_dashboard_ms"] = round(_now_ms() - t0, 3)

            row["active_vehicle_count"] = len(traci.vehicle.getIDList())
            stage["step_profiles"].append(row)

        for key in ("simulation_step_ms", "update_vehicles_ms", "process_charging_ms", "refresh_dashboard_ms"):
            vals = [float(r[key]) for r in stage["step_profiles"]]
            stage[f"avg_{key}"] = round(mean(vals), 3)
            stage[f"max_{key}"] = round(max(vals), 3)

        stage["recommendation_events"] = sum(1 for event in ctrl.charging_events if event.get("event_type") == "recommendation_selected")
        stage["charging_events_total"] = len(ctrl.charging_events)
    finally:
        ctrl.stop()

    return stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Use SUMO-GUI binaries for both diagnostic runs")
    parser.add_argument("--vehicle-count", type=int, default=1000, help="Vehicle count to generate in route file")
    parser.add_argument("--tracked", type=int, default=10, help="Tracked EV count for manifest/controller")
    parser.add_argument("--depart-window", type=int, default=120, help="Departure spread window for generated routes")
    parser.add_argument("--steps", type=int, default=10, help="Number of diagnostic simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for route generation")
    args = parser.parse_args()

    profile: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "use_gui": bool(args.gui),
    }

    t0 = _now_ms()
    build_routes(
        vehicle_count=int(args.vehicle_count),
        tracked_count=int(args.tracked),
        depart_window=int(args.depart_window),
        seed=int(args.seed),
    )
    profile["build_routes_ms"] = round(_now_ms() - t0, 3)

    profile["minimal_no_processing_10_steps"] = run_minimal_10_step(use_gui=bool(args.gui), steps=int(args.steps))
    profile["controller_stage_profile_10_steps"] = run_controller_stage_profile(use_gui=bool(args.gui), steps=int(args.steps))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Wrote diagnostic profile to {OUTPUT}")
    print(json.dumps({
        "build_routes_ms": profile["build_routes_ms"],
        "minimal_known_count": profile["minimal_no_processing_10_steps"]["known_vehicle_count_after_steps"],
        "minimal_seen_vehicle_ids": profile["minimal_no_processing_10_steps"]["seen_vehicle_ids_count"],
        "minimal_moved_count": profile["minimal_no_processing_10_steps"]["moved_unique_count"],
        "ctrl_start_ms": profile["controller_stage_profile_10_steps"]["ctrl_start_ms"],
        "avg_simulation_step_ms": profile["controller_stage_profile_10_steps"]["avg_simulation_step_ms"],
        "avg_update_vehicles_ms": profile["controller_stage_profile_10_steps"]["avg_update_vehicles_ms"],
        "avg_process_charging_ms": profile["controller_stage_profile_10_steps"]["avg_process_charging_ms"],
        "avg_refresh_dashboard_ms": profile["controller_stage_profile_10_steps"]["avg_refresh_dashboard_ms"],
    }, indent=2))


if __name__ == "__main__":
    main()
