from __future__ import annotations

import time

import traci

from src.simulation.controller import SimulationController


def test_tracked_ev_completes_full_charging_lifecycle() -> None:
    controller = SimulationController(
        sumo_cfg="simulations/bangalore/sim.sumocfg",
        db_path="simulations/bangalore/stations.sqlite",
        model_path="runs/ppo_ckpt/ppo_ev_final.zip",
        fleet_size=20,
        station_count=50,
        tracked=20,
        dashboard_state_interval=25,
        dashboard_report_interval=200,
    )

    completed_tracked_vehicle_id = None

    try:
        controller.start()

        for step in range(10000):
            step_started = time.perf_counter()
            traci.simulationStep()
            controller._update_vehicles(step)
            controller._process_charging(step)
            controller._refresh_dashboard()

            if step % 100 == 0 or step < 5:
                tracked_states = []
                for vehicle in controller.vehicle_manager.list_tracked_vehicles():
                    entry = controller.tracked_lifecycle.get(vehicle.vehicle_id)
                    state = entry.get("state") if entry else None
                    tracked_states.append((vehicle.vehicle_id, state, entry.get("charging_start"), entry.get("charging_end"), entry.get("resume_event")))
                print(
                    f"[diag] step={step} elapsed={time.perf_counter() - step_started:.3f}s tracked_states={tracked_states[:3]}"
                )

            for vehicle in controller.vehicle_manager.list_tracked_vehicles():
                entry = controller.tracked_lifecycle.get(vehicle.vehicle_id)
                if not entry:
                    continue
                if entry.get("charging_end") is not None and entry.get("resume_event") is not None:
                    completed_tracked_vehicle_id = vehicle.vehicle_id
                    break
            if completed_tracked_vehicle_id is not None:
                break
    finally:
        controller.stop()

    assert completed_tracked_vehicle_id is not None

    lifecycle_entry = controller.tracked_lifecycle.get(completed_tracked_vehicle_id)
    assert lifecycle_entry is not None

    assert lifecycle_entry.get("initial_battery") is not None
    assert lifecycle_entry.get("minimum_battery") is not None
    assert lifecycle_entry.get("recommendation_time") is not None
    assert lifecycle_entry.get("recommended_station")
    assert lifecycle_entry.get("arrival_time") is not None
    assert lifecycle_entry.get("charging_start") is not None
    assert lifecycle_entry.get("charging_end") is not None
    assert lifecycle_entry.get("final_battery") is not None
    assert lifecycle_entry.get("resume_event") is not None

    assert float(lifecycle_entry["minimum_battery"]) <= float(lifecycle_entry["initial_battery"])
    assert float(lifecycle_entry["final_battery"]) >= float(lifecycle_entry["minimum_battery"])

    queue_time = lifecycle_entry.get("queue_time")
    assert queue_time is None or int(queue_time) >= 0

    event_types_for_vehicle = [
        str(event.get("event_type"))
        for event in controller.charging_events
        if str(event.get("vehicle_id")) == str(completed_tracked_vehicle_id)
    ]
    assert "battery_low" in event_types_for_vehicle
    assert "recommendation_selected" in event_types_for_vehicle
    assert "reroute_to_station" in event_types_for_vehicle
    assert "arrived_at_station" in event_types_for_vehicle
    assert "charging_started" in event_types_for_vehicle
    assert "charging_completed" in event_types_for_vehicle
    assert "resumed_route" in event_types_for_vehicle

    dashboard_state = getattr(controller, "_last_dashboard_state", None)
    assert dashboard_state is not None
    tracked_lifecycle = dashboard_state.get("tracked_lifecycle", [])
    matching = [row for row in tracked_lifecycle if row.get("vehicle_id") == completed_tracked_vehicle_id]
    assert matching
    assert matching[0].get("resume_event") is not None
