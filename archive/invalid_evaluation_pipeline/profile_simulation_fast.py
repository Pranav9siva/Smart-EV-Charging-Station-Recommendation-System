from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import traci

from src.simulation.controller import SimulationController

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "simulations" / "bangalore" / "sim.sumocfg"
DB = ROOT / "data" / "stations" / "stations.sqlite"
MODEL = ROOT / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"
OUT = ROOT / "outputs" / "profile_simulation_fast.json"
MAX_STEPS = 50


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _avg(values: list[float]) -> float:
    return round(mean(values), 3) if values else 0.0


def main() -> None:
    steps = MAX_STEPS
    ctrl = SimulationController(
        sumo_cfg=str(CFG),
        db_path=str(DB),
        model_path=str(MODEL),
        fleet_size=10,
        station_count=0,
        tracked=2,
        use_gui=False,
    )

    rows: list[dict[str, Any]] = []
    simulation_step_ms: list[float] = []
    update_vehicles_ms: list[float] = []
    process_charging_ms: list[float] = []
    refresh_dashboard_ms: list[float] = []
    collect_metrics_ms: list[float] = []
    visualization_publish_ms: list[float] = []
    step_total_ms: list[float] = []

    started = False
    try:
        ctrl.start()
        started = True

        for step in range(steps):
            row: dict[str, Any] = {"step": step}
            t_step = _now_ms()

            t0 = _now_ms()
            traci.simulationStep()
            sim_ms = _now_ms() - t0

            try:
                ctrl.current_step = int(traci.simulation.getTime())
            except Exception:
                ctrl.current_step = step

            t0 = _now_ms()
            ctrl._update_vehicles(step)
            update_ms = _now_ms() - t0

            t0 = _now_ms()
            ctrl._process_charging(step)
            process_ms = _now_ms() - t0

            t0 = _now_ms()
            ctrl._refresh_dashboard()
            refresh_ms = _now_ms() - t0

            t0 = _now_ms()
            ctrl._collect_metrics(step)
            collect_ms = _now_ms() - t0

            viz_ms = 0.0
            if getattr(ctrl, "visualization", None) is not None:
                t0 = _now_ms()
                ctrl.visualization.publish_step(step)
                viz_ms = _now_ms() - t0

            total_ms = _now_ms() - t_step

            active_count = 0
            try:
                active_count = len(traci.vehicle.getIDList())
            except Exception:
                active_count = 0

            row.update(
                {
                    "simulation_step_ms": round(sim_ms, 3),
                    "update_vehicles_ms": round(update_ms, 3),
                    "process_charging_ms": round(process_ms, 3),
                    "refresh_dashboard_ms": round(refresh_ms, 3),
                    "collect_metrics_ms": round(collect_ms, 3),
                    "visualization_publish_ms": round(viz_ms, 3),
                    "step_total_ms": round(total_ms, 3),
                    "active_vehicle_count": int(active_count),
                }
            )
            rows.append(row)

            simulation_step_ms.append(sim_ms)
            update_vehicles_ms.append(update_ms)
            process_charging_ms.append(process_ms)
            refresh_dashboard_ms.append(refresh_ms)
            collect_metrics_ms.append(collect_ms)
            visualization_publish_ms.append(viz_ms)
            step_total_ms.append(total_ms)

            if (step + 1) % 10 == 0:
                print(
                    f"step={step + 1} "
                    f"sim={sim_ms:.3f}ms update={update_ms:.3f}ms process={process_ms:.3f}ms "
                    f"dashboard={refresh_ms:.3f}ms metrics={collect_ms:.3f}ms viz={viz_ms:.3f}ms total={total_ms:.3f}ms"
                )

        result = {
            "steps": steps,
            "fleet_size": 10,
            "tracked": 2,
            "sumo_cfg": str(CFG),
            "avg_simulation_step_ms": _avg(simulation_step_ms),
            "avg_update_vehicles_ms": _avg(update_vehicles_ms),
            "avg_process_charging_ms": _avg(process_charging_ms),
            "avg_refresh_dashboard_ms": _avg(refresh_dashboard_ms),
            "avg_collect_metrics_ms": _avg(collect_metrics_ms),
            "avg_visualization_publish_ms": _avg(visualization_publish_ms),
            "avg_step_total_ms": _avg(step_total_ms),
            "max_step_total_ms": round(max(step_total_ms), 3) if step_total_ms else 0.0,
            "rows": rows,
        }

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print("Average timings (ms):")
        print(json.dumps({k: v for k, v in result.items() if k.startswith("avg_") or k == "max_step_total_ms"}, indent=2))
        print(f"Wrote fast profile to {OUT}")
    finally:
        if started:
            ctrl.stop()


if __name__ == "__main__":
    main()
