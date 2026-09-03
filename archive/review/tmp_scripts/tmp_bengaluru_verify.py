import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import traci
from src.simulation.controller import SimulationController

print('closing any existing traci connection')
try:
    traci.close()
except Exception as exc:
    print('traci.close failed', exc)

ctrl = SimulationController(
    'simulations/bangalore/sim.sumocfg',
    'runs/tmp_stations.db',
    'runs/ppo_ckpt/ppo_ev_final.zip',
    fleet_size=10,
    station_count=50,
    tracked=10,
    charge_threshold_pct=20.0,
)

print('starting controller')
ctrl.start()
print('started controller')
print('vehicle_manager vehicles created', len(ctrl.vehicle_manager.list_vehicles()))
print('vm_to_sumo', ctrl.vm_to_sumo)
print('sumo_to_vm', ctrl.sumo_to_vm)
try:
    print('sumo ids after start', traci.vehicle.getIDList())
except Exception as exc:
    print('sumo ids after start error', exc)

print('running 500 steps')
ctrl.run(steps=500)
print('completed 500 steps')
print('recommendation_log rows', len(ctrl.recommendation_log))
print('charging events', len(ctrl.charging_events))
print('simulation metrics', len(ctrl.simulation_metrics))
print('reward curve', len(ctrl.reward_curve))
print('assignments', list(ctrl.assignments.keys()))
