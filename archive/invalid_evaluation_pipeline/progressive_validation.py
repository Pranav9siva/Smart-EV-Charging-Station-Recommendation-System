from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.controller import SimulationController


CASES: list[dict[str, Any]] = [
    {"name": "TEST 1", "fleet_size": 5, "steps": 10, "timeout_seconds": 90},
    {"name": "TEST 2", "fleet_size": 20, "steps": 20, "timeout_seconds": 90},
    {"name": "TEST 3", "fleet_size": 100, "steps": 50, "timeout_seconds": 180},
    {"name": "TEST 4", "fleet_size": 500, "steps": 100, "timeout_seconds": 240},
    {"name": "TEST 5", "fleet_size": 1000, "steps": 200, "timeout_seconds": 300},
]

QUICK_CASE: dict[str, Any] = {"name": "QUICK", "fleet_size": 20, "steps": 20, "timeout_seconds": 120, "tracked": 5, "use_gui": False}


class ValidationController(SimulationController):
    def __init__(self, *args: Any, status_path: Path | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.status_path = status_path
        self.validation_state: dict[str, Any] = {
            "last_stage": "initialized",
            "vehicle_position_updates": 0,
            "dashboard_updates": 0,
            "station_data_available": False,
            "traffic_data_available": False,
            "errors": [],
        }
        self._write_status("initialized")

    def _write_status(self, stage: str, step: int | None = None) -> None:
        if self.status_path is None:
            return
        payload = {
            "last_stage": stage,
            "step": self.current_step if step is None else step,
            "vehicle_position_updates": self.validation_state.get("vehicle_position_updates", 0),
            "dashboard_updates": self.validation_state.get("dashboard_updates", 0),
        }
        try:
            self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _update_vehicles(self, step: int) -> None:
        self.validation_state["last_stage"] = "update_vehicles"
        self.validation_state["vehicle_position_updates"] += 1
        self._write_status("update_vehicles", step)
        super()._update_vehicles(step)

    def _process_charging(self, step: int) -> None:
        self.validation_state["last_stage"] = "process_charging"
        self._write_status("process_charging", step)
        super()._process_charging(step)

    def _collect_metrics(self, step: int) -> None:
        self.validation_state["last_stage"] = "collect_metrics"
        self._write_status("collect_metrics", step)
        super()._collect_metrics(step)

    def _refresh_dashboard(self) -> None:
        self.validation_state["last_stage"] = "refresh_dashboard"
        self.validation_state["dashboard_updates"] += 1
        self._write_status("refresh_dashboard", self.current_step)
        super()._refresh_dashboard()
        last_state = getattr(self, "_last_dashboard_state", {}) or {}
        station_details = last_state.get("station_details") or []
        traffic = last_state.get("traffic") or []
        self.validation_state["station_data_available"] = bool(station_details)
        self.validation_state["traffic_data_available"] = bool(traffic)


def _terminate_sumo_and_python(process: subprocess.Popen[str] | None = None) -> None:
    try:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    except Exception:
        pass

    for image_name in ("sumo.exe", "sumo-gui.exe"):
        try:
            subprocess.run(["taskkill", "/F", "/IM", image_name, "/T"], capture_output=True, text=True, timeout=10)
        except Exception:
            pass

    current_pid = os.getpid()
    try:
        subprocess.run(["taskkill", "/F", "/PID", str(current_pid), "/T"], capture_output=True, text=True, timeout=10)
    except Exception:
        pass


def _get_cases(quick: bool) -> list[dict[str, Any]]:
    if quick:
        return [QUICK_CASE]
    return CASES


def run_case(case: dict[str, Any], status_path: Path, result_path: Path) -> None:
    start_time = time.perf_counter()
    errors: list[str] = []
    controller: ValidationController | None = None
    initial_battery: dict[str, float] = {}
    try:
        tracked = int(case.get("tracked", max(1, min(int(case["fleet_size"]), 10))))
        controller = ValidationController(
            sumo_cfg=str(ROOT / "simulations" / "bangalore" / "sim.sumocfg"),
            db_path=str(ROOT / "data" / "stations" / "stations.sqlite"),
            model_path=str(ROOT / "runs" / "ppo_ckpt" / "ppo_ev_final.zip"),
            fleet_size=int(case["fleet_size"]),
            tracked=tracked,
            use_gui=bool(case.get("use_gui", False)),
            dashboard_state_interval=5,
            dashboard_report_interval=10,
            visualization_interval=5,
            metrics_interval=5,
            status_path=status_path,
        )
        controller._write_status("starting")
        print(f"[{case['name']}] Starting validation")
        controller.start()
        print(f"[{case['name']}] SUMO started")
        initial_battery = {
            str(getattr(vehicle, "vehicle_id", "")): float(getattr(vehicle, "battery_pct", 0.0) or 0.0)
            for vehicle in controller.vehicle_manager.list_vehicles()
        }
        controller._write_status("started")
        controller.run(steps=int(case["steps"]))
        controller._write_status("completed")
        print(f"[{case['name']}] Simulation completed")
    except Exception as exc:  # pragma: no cover - surfaced in validation report
        errors.append(f"{type(exc).__name__}: {exc}")
        try:
            controller._write_status("error")
        except Exception:
            pass
        print(f"[{case['name']}] Validation error: {exc}")
    finally:
        try:
            if controller is not None:
                controller.stop()
        except Exception:
            pass

    try:
        vehicles = list(getattr(controller, "vehicle_manager", None).list_vehicles()) if controller is not None else []
    except Exception:
        vehicles = []

    final_battery = {
        str(getattr(vehicle, "vehicle_id", "")): float(getattr(vehicle, "battery_pct", 0.0) or 0.0)
        for vehicle in vehicles
    }

    last_dashboard_state = getattr(controller, "_last_dashboard_state", None) if controller is not None else None
    station_details = last_dashboard_state.get("station_details") if isinstance(last_dashboard_state, dict) else []
    traffic = last_dashboard_state.get("traffic") if isinstance(last_dashboard_state, dict) else []

    battery_decrease = sum(max(0.0, initial_battery.get(vehicle_id, 0.0) - final_battery.get(vehicle_id, 0.0)) for vehicle_id in initial_battery)
    battery_increase = sum(max(0.0, final_battery.get(vehicle_id, 0.0) - initial_battery.get(vehicle_id, 0.0)) for vehicle_id in initial_battery)

    result = {
        "name": case["name"],
        "fleet_size": int(case["fleet_size"]),
        "steps": int(case["steps"]),
        "execution_time_seconds": round(time.perf_counter() - start_time, 3),
        "vehicles_created": len(vehicles),
        "tracked_vehicles": len(controller.vehicle_manager.list_tracked_vehicles()) if controller is not None else 0,
        "simulation_steps": int(getattr(controller, "current_step", 0)) if controller is not None else 0,
        "ppo_decisions": int(getattr(controller, "rec_call_count", 0)) if controller is not None else 0,
        "recommendations": int(len(getattr(controller, "recommendation_log", []))) if controller is not None else 0,
        "charging_events": int(len(getattr(controller, "charging_events", []))) if controller is not None else 0,
        "queue_events": int(sum(1 for event in getattr(controller, "charging_events", []) if str(event.get("event_type", "")).startswith("queue"))) if controller is not None else 0,
        "battery_decrease": round(battery_decrease, 3),
        "battery_increase": round(battery_increase, 3),
        "dashboard_updates": int(controller.validation_state.get("dashboard_updates", 0)) if controller is not None else 0,
        "vehicle_position_updates": int(controller.validation_state.get("vehicle_position_updates", 0)) if controller is not None else 0,
        "station_data_availability": {"available": bool(station_details), "count": len(station_details or [])},
        "traffic_data_availability": {"available": bool(traffic), "count": len(traffic or [])},
        "errors": errors,
        "last_stage": controller.validation_state.get("last_stage") if controller is not None else "not_started",
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{case['name']}] Dashboard state generated")
    print(f"[{case['name']}] Validation completed")


def _write_report(output_dir: Path, results: list[dict[str, Any]]) -> None:
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps({"cases": results}, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", default=None)
    parser.add_argument("--result-path", default=None)
    parser.add_argument("--status-path", default=None)
    parser.add_argument("--quick", action="store_true", help="Run a single fast headless validation case")
    parser.add_argument("--subprocess-case", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.case_name:
        case = next((item for item in _get_cases(args.quick) if item["name"].lower() == args.case_name.lower()), None)
        if case is None:
            raise SystemExit(f"Unknown case {args.case_name}")
        result_path = Path(args.result_path or output_dir / f"{case['name'].lower().replace(' ', '_')}_result.json")
        status_path = Path(args.status_path or output_dir / f"{case['name'].lower().replace(' ', '_')}_status.json")
        status_path.parent.mkdir(parents=True, exist_ok=True)

        if not args.subprocess_case:
            timeout_seconds = max(1, int(case.get("timeout_seconds", 120)))
            print(f"[1/1] Starting quick validation..." if args.quick else f"[1/1] Starting {case['name']} validation...")
            command = [sys.executable, str(ROOT / "scripts" / "progressive_validation.py"), "--case-name", case["name"], "--result-path", str(result_path), "--status-path", str(status_path), "--subprocess-case"]
            if args.quick:
                command.append("--quick")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            try:
                subprocess.run(command, cwd=str(ROOT), env=env, timeout=float(timeout_seconds))
            except subprocess.TimeoutExpired:
                _terminate_sumo_and_python()
                timed_out_result = {
                    "name": case["name"],
                    "fleet_size": int(case["fleet_size"]),
                    "steps": int(case["steps"]),
                    "execution_time_seconds": float(timeout_seconds),
                    "vehicles_created": 0,
                    "tracked_vehicles": 0,
                    "simulation_steps": 0,
                    "ppo_decisions": 0,
                    "recommendations": 0,
                    "charging_events": 0,
                    "queue_events": 0,
                    "battery_decrease": 0.0,
                    "battery_increase": 0.0,
                    "dashboard_updates": 0,
                    "vehicle_position_updates": 0,
                    "station_data_availability": {"available": False, "count": 0},
                    "traffic_data_availability": {"available": False, "count": 0},
                    "errors": [f"timed_out_after_{timeout_seconds}_seconds"],
                    "last_stage": "timeout",
                }
                result_path.write_text(json.dumps(timed_out_result, indent=2), encoding="utf-8")
                _write_report(output_dir, [timed_out_result])
                print(f"[{case['name']}] Validation timed out")
                return 1
            return 0

        run_case(case, status_path, result_path)
        return 0

    cases = _get_cases(args.quick)
    results: list[dict[str, Any]] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        case_name = case["name"]
        status_path = output_dir / f"{case_name.lower().replace(' ', '_')}_status.json"
        result_path = output_dir / f"{case_name.lower().replace(' ', '_')}_result.json"
        print(f"[{index}/{total}] Starting {case_name} validation...")
        command = [sys.executable, str(ROOT / "scripts" / "progressive_validation.py"), "--case-name", case_name, "--result-path", str(result_path), "--status-path", str(status_path), "--subprocess-case"]
        if args.quick:
            command.append("--quick")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            subprocess.run(command, cwd=str(ROOT), env=env, timeout=float(case.get("timeout_seconds", 120)))
        except subprocess.TimeoutExpired:
            _terminate_sumo_and_python()
            timed_out_result = {
                "name": case_name,
                "fleet_size": int(case["fleet_size"]),
                "steps": int(case["steps"]),
                "execution_time_seconds": float(case.get("timeout_seconds", 120)),
                "vehicles_created": 0,
                "tracked_vehicles": 0,
                "simulation_steps": 0,
                "ppo_decisions": 0,
                "recommendations": 0,
                "charging_events": 0,
                "queue_events": 0,
                "battery_decrease": 0.0,
                "battery_increase": 0.0,
                "dashboard_updates": 0,
                "vehicle_position_updates": 0,
                "station_data_availability": {"available": False, "count": 0},
                "traffic_data_availability": {"available": False, "count": 0},
                "errors": [f"timed_out_after_{case.get('timeout_seconds', 120)}_seconds"],
                "last_stage": "timeout",
            }
            result_path.write_text(json.dumps(timed_out_result, indent=2), encoding="utf-8")
            results.append(timed_out_result)
            _write_report(output_dir, results)
            print(f"[{index}/{total}] {case_name}: TIMED OUT")
            continue

        if result_path.exists():
            try:
                results.append(json.loads(result_path.read_text(encoding="utf-8")))
            except Exception:
                results.append({"name": case_name, "errors": ["failed_to_read_result"]})
        else:
            results.append({"name": case_name, "errors": ["result_file_missing"]})

        _write_report(output_dir, results)
        print(f"[{index}/{total}] {case_name}: completed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
