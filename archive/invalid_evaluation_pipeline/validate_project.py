"""Lightweight project validation script.

This script does not run long simulations. It validates that core runtime
artifacts and imports needed by the EV simulation stack are present.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_path(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "is_file": path.is_file(),
    }


def check_import(module_name: str) -> dict[str, object]:
    try:
        importlib.import_module(module_name)
        return {"module": module_name, "ok": True, "error": None}
    except Exception as exc:  # pragma: no cover
        return {"module": module_name, "ok": False, "error": str(exc)}


def main() -> int:
    required_files = [
        ROOT / "scripts" / "run_simulation.py",
        ROOT / "simulations" / "bangalore" / "sim.sumocfg",
        ROOT / "simulations" / "bangalore" / "network.net.xml",
        ROOT / "simulations" / "bangalore" / "evs.rou.xml",
        ROOT / "simulations" / "bangalore" / "stations.add.xml",
        ROOT / "data" / "stations" / "stations.sqlite",
        ROOT / "runs" / "ppo_ckpt" / "ppo_ev_final.zip",
    ]

    imports = [
        "src.simulation.controller",
        "src.station_management.manager",
        "src.ev_management.vehicle_manager",
        "src.recommendation_engine.engine",
        "src.rl_env.gym_ev_charging_env",
        "src.visualization.frontend.app",
        "src.xai.explainer",
    ]

    file_results = [check_path(path) for path in required_files]
    import_results = [check_import(name) for name in imports]

    report = {
        "required_files": file_results,
        "imports": import_results,
    }

    out_path = ROOT / "outputs" / "reports" / "validation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    files_ok = all(item["exists"] and item["is_file"] for item in file_results)
    imports_ok = all(item["ok"] for item in import_results)

    print(json.dumps(report, indent=2))
    if files_ok and imports_ok:
        print("VALIDATION_STATUS: PASS")
        return 0
    print("VALIDATION_STATUS: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
