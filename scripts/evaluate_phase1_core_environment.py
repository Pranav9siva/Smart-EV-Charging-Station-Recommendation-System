from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import traci

from src.simulation.controller import SimulationController

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "runs" / "evaluation" / "phase1_core_environment"
EVENTS_CSV = OUTPUT_DIR / "phase1_events.csv"
RESULT_JSON = OUTPUT_DIR / "phase1_core_environment.json"
SUMMARY_MD = OUTPUT_DIR / "phase1_core_environment_summary.md"


@dataclass
class LifecycleScenarioResult:
    name: str
    fleet_size: int
    station_count: int
    tracked: int
    seed: int
    steps_ran: int
    checks: dict[str, bool]
    evidence: dict[str, Any]


def _find_first(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for event in events:
        if str(event.get("event_type")) == event_type:
            return event
    return None


def _event_types(events: list[dict[str, Any]]) -> set[str]:
    return {str(event.get("event_type") or "") for event in events}


def _run_scenario(
    *,
    name: str,
    fleet_size: int,
    station_count: int,
    tracked: int,
    seed: int,
    max_steps: int,
    low_soc_vehicle_count: int,
) -> tuple[LifecycleScenarioResult, list[dict[str, Any]]]:
    scenario_db_path = OUTPUT_DIR / f"{name}_stations.sqlite"
    scenario_db_path.parent.mkdir(parents=True, exist_ok=True)
    if scenario_db_path.exists():
        scenario_db_path.unlink()

    controller = SimulationController(
        sumo_cfg="simulations/bangalore/sim.sumocfg",
        db_path=str(scenario_db_path),
        model_path="runs/ppo_ckpt/ppo_ev_final.zip",
        fleet_size=fleet_size,
        station_count=station_count,
        tracked=tracked,
        charge_threshold_pct=20.0,
        use_gui=False,
        simulation_seed=seed,
    )

    checks = {
        "battery_low_detected": False,
        "recommendation_selected": False,
        "rerouted_to_station": False,
        "arrived_at_station": False,
        "charging_started": False,
        "charging_progress": False,
        "charging_completed": False,
        "resumed_route": False,
    }

    steps_ran = 0
    scenario_events: list[dict[str, Any]] = []

    try:
        controller.start()

        # Force a realistic low-SOC trigger for a subset of vehicles so charging lifecycle
        # is exercised deterministically.
        low_soc_targets = controller.vehicle_manager.list_vehicles()[: max(1, low_soc_vehicle_count)]
        for vehicle in low_soc_targets:
            vehicle.battery_pct = 8.0
            vehicle.remaining_range_km = 2.0

        for step in range(max_steps):
            try:
                if traci.simulation.getMinExpectedNumber() <= 0 and not controller.assignments:
                    break
            except Exception:
                break

            before = len(controller.charging_events)
            try:
                traci.simulationStep()
            except Exception:
                break
            controller._update_vehicles(step)
            controller._process_charging(step)

            new_events = controller.charging_events[before:]
            for event in new_events:
                event_copy = dict(event)
                event_copy["scenario"] = name
                scenario_events.append(event_copy)

            seen_types = _event_types(scenario_events)
            checks["battery_low_detected"] = "battery_low" in seen_types
            checks["recommendation_selected"] = "recommendation_selected" in seen_types
            checks["rerouted_to_station"] = "reroute_to_station" in seen_types
            checks["arrived_at_station"] = "arrived_at_station" in seen_types
            checks["charging_started"] = "charging_started" in seen_types
            checks["charging_progress"] = "charging_progress" in seen_types
            checks["charging_completed"] = "charging_completed" in seen_types
            checks["resumed_route"] = "resumed_route" in seen_types

            steps_ran = step + 1
            if all(checks.values()):
                break

        first_started = _find_first(scenario_events, "charging_started")
        first_completed = _find_first(scenario_events, "charging_completed")
        first_resumed = _find_first(scenario_events, "resumed_route")

        unique_charge_or_queue = {
            str(event.get("vehicle_id"))
            for event in scenario_events
            if str(event.get("event_type")) in {"charging_started", "queue_joined"}
        }

        evidence: dict[str, Any] = {
            "first_charging_started": first_started,
            "first_charging_completed": first_completed,
            "first_resumed_route": first_resumed,
            "unique_vehicles_with_charging_or_queue": sorted(v for v in unique_charge_or_queue if v),
            "charging_or_queue_vehicle_count": len([v for v in unique_charge_or_queue if v]),
            "low_soc_targets": [v.vehicle_id for v in low_soc_targets],
        }

        if name == "tiny_1ev_2stations":
            soc_before = first_started.get("battery_pct") if first_started else None
            soc_after = first_completed.get("battery_pct") if first_completed else None
            evidence["soc_before_charging"] = soc_before
            evidence["soc_after_charging"] = soc_after
            evidence["soc_increased"] = (
                soc_before is not None
                and soc_after is not None
                and float(soc_after) >= float(soc_before)
            )
            checks["soc_increased"] = bool(evidence["soc_increased"])

        if name == "medium_10ev_20stations":
            checks["multiple_independent_charging_or_queue"] = len([v for v in unique_charge_or_queue if v]) >= 2

        return (
            LifecycleScenarioResult(
                name=name,
                fleet_size=fleet_size,
                station_count=station_count,
                tracked=tracked,
                seed=seed,
                steps_ran=steps_ran,
                checks=checks,
                evidence=evidence,
            ),
            scenario_events,
        )
    finally:
        controller.stop()


def _write_outputs(results: list[LifecycleScenarioResult], events: list[dict[str, Any]]) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with EVENTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scenario",
            "event_type",
            "step",
            "vehicle_id",
            "station_id",
            "port_id",
            "battery_pct",
            "queue_position",
            "destination_edge",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow({
                "scenario": event.get("scenario"),
                "event_type": event.get("event_type"),
                "step": event.get("step"),
                "vehicle_id": event.get("vehicle_id"),
                "station_id": event.get("station_id"),
                "port_id": event.get("port_id"),
                "battery_pct": event.get("battery_pct"),
                "queue_position": event.get("queue_position"),
                "destination_edge": event.get("destination_edge"),
            })

    payload = {
        "phase": "PHASE_1_CORE_SIMULATION_STABILIZATION",
        "output_dir": str(OUTPUT_DIR),
        "scenarios": [
            {
                "name": result.name,
                "fleet_size": result.fleet_size,
                "station_count": result.station_count,
                "tracked": result.tracked,
                "seed": result.seed,
                "steps_ran": result.steps_ran,
                "checks": result.checks,
                "evidence": result.evidence,
                "passed": all(result.checks.values()),
            }
            for result in results
        ],
    }
    payload["complete_ev_lifecycle_passed"] = all(s["passed"] for s in payload["scenarios"])

    RESULT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Phase 1 Core Environment Evaluation",
        "",
        f"- Complete EV lifecycle passed: {'YES' if payload['complete_ev_lifecycle_passed'] else 'NO'}",
        "- Required lifecycle: EV start -> SOC drop -> charging threshold -> recommendation -> reroute -> station occupancy update -> charging progress -> SOC increase -> charging completion -> resume route.",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Fleet | Stations | Seed | Passed | Steps |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for scenario in payload["scenarios"]:
        lines.append(
            f"| {scenario['name']} | {scenario['fleet_size']} | {scenario['station_count']} | {scenario['seed']} | {'YES' if scenario['passed'] else 'NO'} | {scenario['steps_ran']} |"
        )

    lines.extend(["", "## Lifecycle Check Matrix", ""])
    for scenario in payload["scenarios"]:
        lines.append(f"### {scenario['name']}")
        for key, value in scenario["checks"].items():
            lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
        lines.append("")

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    return payload


def run_phase1_core_environment() -> dict[str, Any]:
    scenarios: list[LifecycleScenarioResult] = []
    all_events: list[dict[str, Any]] = []

    tiny_result, tiny_events = _run_scenario(
        name="tiny_1ev_2stations",
        fleet_size=1,
        station_count=2,
        tracked=1,
        seed=101,
        max_steps=7000,
        low_soc_vehicle_count=1,
    )
    scenarios.append(tiny_result)
    all_events.extend(tiny_events)

    medium_result, medium_events = _run_scenario(
        name="medium_10ev_20stations",
        fleet_size=10,
        station_count=20,
        tracked=10,
        seed=202,
        max_steps=9000,
        low_soc_vehicle_count=5,
    )
    scenarios.append(medium_result)
    all_events.extend(medium_events)

    return _write_outputs(scenarios, all_events)


if __name__ == "__main__":
    run_phase1_core_environment()
