from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import traci

from src.simulation.controller import SimulationController


@dataclass
class LifecycleRunResult:
    status: dict[str, bool]
    first_started: dict | None
    first_completed: dict | None
    first_resumed: dict | None
    queue_events: list[dict]
    started_events: list[dict]


def _run_lifecycle(steps: int) -> LifecycleRunResult:
    controller = SimulationController(
        sumo_cfg="simulations/bangalore/sim.sumocfg",
        db_path="simulations/bangalore/stations.sqlite",
        model_path="runs/ppo_ckpt/ppo_ev_final.zip",
        fleet_size=20,
        station_count=50,
        tracked=5,
    )

    status = {
        "battery_low": False,
        "recommendation_selected": False,
        "reroute_to_station": False,
        "arrived_at_station": False,
        "queue_joined": False,
        "charging_started": False,
        "charging_progress": False,
        "charging_completed": False,
        "resumed_route": False,
    }
    first_started = None
    first_completed = None
    first_resumed = None
    queue_events: list[dict] = []
    started_events: list[dict] = []

    try:
        controller.start()

        for step in range(steps):
            before = len(controller.charging_events)
            traci.simulationStep()
            controller._update_vehicles(step)
            controller._process_charging(step)

            for event in controller.charging_events[before:]:
                event_type = str(event.get("event_type") or "")
                if event_type in status:
                    status[event_type] = True
                if event_type == "queue_joined":
                    queue_events.append(dict(event))
                elif event_type == "charging_started":
                    started_events.append(dict(event))
                    if first_started is None:
                        first_started = dict(event)
                elif event_type == "charging_completed" and first_completed is None:
                    first_completed = dict(event)
                elif event_type == "resumed_route" and first_resumed is None:
                    first_resumed = dict(event)

            if steps > 500 and status["charging_completed"] and status["resumed_route"]:
                break
    finally:
        controller.stop()

    return LifecycleRunResult(
        status=status,
        first_started=first_started,
        first_completed=first_completed,
        first_resumed=first_resumed,
        queue_events=queue_events,
        started_events=started_events,
    )


def test_smoke_charging_lifecycle_700_steps() -> None:
    result = _run_lifecycle(steps=700)

    assert result.status["battery_low"]
    assert result.status["recommendation_selected"]
    assert result.status["reroute_to_station"]
    assert result.status["arrived_at_station"]
    assert result.status["charging_started"]
    assert result.status["charging_progress"]


def test_long_horizon_charging_lifecycle_9000_steps() -> None:
    result = _run_lifecycle(steps=9000)

    assert result.status["charging_started"]
    assert result.status["charging_progress"]
    assert result.status["charging_completed"]
    assert result.status["resumed_route"]

    assert result.first_started is not None
    assert result.first_completed is not None
    assert result.first_resumed is not None

    started_battery = result.first_started.get("battery_pct")
    completed_battery = result.first_completed.get("battery_pct")
    assert started_battery is not None
    assert completed_battery is not None
    assert float(completed_battery) >= float(started_battery)

    completed_station = str(result.first_completed.get("station_id") or "")
    completed_port = str(result.first_completed.get("port_id") or "")
    assert completed_station
    assert completed_port

    controller_for_state = SimulationController(
        sumo_cfg="simulations/bangalore/sim.sumocfg",
        db_path="simulations/bangalore/stations.sqlite",
        model_path="runs/ppo_ckpt/ppo_ev_final.zip",
        fleet_size=20,
        station_count=50,
        tracked=5,
    )
    try:
        station_state = controller_for_state.station_manager.get_station_state(completed_station) or {}
        ports = station_state.get("ports", [])
        matching_port = next((port for port in ports if port.get("port_id") == completed_port), None)
        assert matching_port is not None

        # Port may be immediately reused by queued vehicles; ensure Available transition was persisted.
        available_events = controller_for_state.station_manager.repo.connection.execute(
            "SELECT COUNT(*) FROM occupancy_log WHERE port_id = ? AND status = 'Available'",
            (completed_port,),
        ).fetchone()
        assert available_events is not None
        assert int(available_events[0]) >= 1
    finally:
        controller_for_state.station_manager.close()

    if result.queue_events:
        queue_steps_by_vehicle = defaultdict(list)
        for event in result.queue_events:
            queue_steps_by_vehicle[str(event.get("vehicle_id"))].append(int(event.get("step", -1)))

        handoff_observed = False
        for started in result.started_events:
            vehicle_id = str(started.get("vehicle_id"))
            started_step = int(started.get("step", -1))
            queued_steps = queue_steps_by_vehicle.get(vehicle_id, [])
            if any(step < started_step for step in queued_steps):
                handoff_observed = True
                break
        assert handoff_observed
