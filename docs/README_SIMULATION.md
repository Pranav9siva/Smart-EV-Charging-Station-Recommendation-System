# Example SUMO simulation

Files are under `simulations/example/`.

Quick run (ensure SUMO is in `SUMO_HOME` or `PATH` and venv is activated):

```powershell
# generate network from nodes/edges
netconvert --node-files simulations/example/nodes.xml --edge-files simulations/example/edges.xml -o simulations/example/sim.net.xml

# run headless via TraCI
.venv\Scripts\python.exe scripts\run_traci.py

# or run GUI
sumo-gui -c simulations/example/sim.sumocfg
```
## Bangalore scenario

The Bangalore demo can be generated and run with the SUMO GUI via:

```powershell
.venv\Scripts\python.exe scripts/run_traci_bangalore.py
```
