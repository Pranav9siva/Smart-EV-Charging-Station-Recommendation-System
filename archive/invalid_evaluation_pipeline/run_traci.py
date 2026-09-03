import os
from pathlib import Path
from sumolib import checkBinary
import traci

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / 'simulations' / 'example'
CFG = EX / 'sim.sumocfg'

def main():
    sumo_binary = checkBinary('sumo')
    print('Using SUMO binary:', sumo_binary)
    traci.start([sumo_binary, '-c', str(CFG)])
    step = 0
    try:
        while step < 100:
            traci.simulationStep()
            v = traci.vehicle.getIDList()
            if v:
                print('step', step, 'vehicles', v)
            step += 1
    finally:
        traci.close()

if __name__ == '__main__':
    main()
