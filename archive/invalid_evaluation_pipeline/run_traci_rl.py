import time
from pathlib import Path
from sumolib import checkBinary
import traci

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / 'simulations' / 'expanded'
CFG = EX / 'sim.sumocfg'

def run_episode(ep):
    sumo_binary = checkBinary('sumo')
    print(f'episode {ep}: starting SUMO ({sumo_binary})')
    traci.start([sumo_binary, '-c', str(CFG), '--start', '--quit-on-end'])
    step = 0
    try:
        while step < 200:
            traci.simulationStep()
            if step % 20 == 0:
                vehs = traci.vehicle.getIDList()
                tls = traci.trafficlight.getIDList()
                print(f'ep{ep} step{step} vehs={vehs} tls={tls}')
            step += 1
    finally:
        traci.close()

def main():
    episodes = 3
    for ep in range(episodes):
        run_episode(ep)
        time.sleep(0.2)

if __name__ == '__main__':
    main()
