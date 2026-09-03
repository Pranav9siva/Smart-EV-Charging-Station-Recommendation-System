# Environment Setup — RL-Based EV Charging Recommendation System

This guide sets up everything your project doc describes: Python 3.12, VS Code, Git,
SUMO + TraCI, OSM → SUMO network conversion, Gymnasium, Stable-Baselines3 (PyTorch),
and SQLite/PostgreSQL. Instructions are given per OS — skip to the section you need.

---

## 0. Recommended project folder structure

Create this now so every module in your design has a home:

```
ev-charging-rl/
├── venv/                     # Python virtual environment (not committed to git)
├── sumo/
│   ├── network/              # .net.xml, .osm files, netconvert configs
│   ├── routes/                # .rou.xml vehicle/route definitions
│   ├── stations/              # charging station additional files (.add.xml)
│   └── config/                # .sumocfg files
├── src/
│   ├── simulation/            # TraCI <-> SUMO communication module
│   ├── rl_env/                # Gymnasium environment (state/action/reward)
│   ├── route_planning/        # NetworkX-based route/graph module
│   ├── station_management/    # port availability, waiting time, grid load
│   └── recommendation_engine/ # combines everything -> station recommendation
├── data/
│   ├── db.sqlite3
│   └── logs/
├── notebooks/                 # experiments, training curves
├── requirements.txt
├── .gitignore
└── README.md
```

```bash
mkdir -p ev-charging-rl/{sumo/{network,routes,stations,config},src/{simulation,rl_env,route_planning,station_management,recommendation_engine},data/logs,notebooks}
cd ev-charging-rl
git init
```

---

## 1. Install Python 3.12

**Windows**
- Download from https://www.python.org/downloads/ (check "Add python.exe to PATH" during install).
- Verify: `python --version`

**macOS**
```bash
brew install python@3.12
python3.12 --version
```

**Linux (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip -y
python3.12 --version
```

---

## 2. Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# macOS / Linux
python3.12 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt from here on.

---

## 3. Install VS Code + extensions

1. Download VS Code: https://code.visualstudio.com/
2. Install extensions: **Python** (Microsoft), **Pylance**, **Jupyter**, **GitLens** (optional).
3. In VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose the `venv` you just created.

---

## 4. Git + GitHub

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
Create a repo on GitHub, then:
```bash
git remote add origin https://github.com/<your-username>/ev-charging-rl.git
```
Add a `.gitignore`:
```
venv/
__pycache__/
*.pyc
data/*.sqlite3
*.log
.env
```

---

## 5. Install SUMO

SUMO is a standalone application — it is **not** a `pip install`. TraCI connects
Python to a running SUMO process, so SUMO itself must be installed separately.

**Windows**
1. Download the installer from https://sumo.dlr.de/docs/Downloads.php (or `winget install DLR-TS.SUMO`).
2. Note the install path, e.g. `C:\Program Files (x86)\Eclipse\Sumo`.
3. Set the `SUMO_HOME` environment variable to that path:
   - Search "Environment Variables" in Windows settings → New system variable
   - Name: `SUMO_HOME`, Value: `C:\Program Files (x86)\Eclipse\Sumo`
4. Add `%SUMO_HOME%\bin` to your `PATH`.

**macOS**
```bash
brew tap dlr-ts/sumo
brew install sumo
echo 'export SUMO_HOME="/opt/homebrew/opt/sumo/share/sumo"' >> ~/.zshrc
source ~/.zshrc
```

**Linux (Ubuntu/Debian)**
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt update
sudo apt install sumo sumo-tools sumo-doc -y
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```

**Verify (all OS):**
```bash
sumo --version
echo $SUMO_HOME      # or echo %SUMO_HOME% on Windows
```

---

## 6. Install Python packages

With the venv active and in your project root:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
(The `requirements.txt` alongside this guide already pins gymnasium, stable-baselines3,
torch, traci, sumolib, numpy, pandas, networkx, matplotlib, plotly, sqlalchemy, psycopg2-binary.)

Verify TraCI can see SUMO:
```python
import traci
import sumolib
print("traci + sumolib import OK")
```

---

## 7. Import a real road network from OpenStreetMap

1. Go to https://www.openstreetmap.org/export (or use https://extract.bbbike.org/ for a larger custom area) and download the `.osm` file for your chosen city/region — keep the area small at first (a few km²) so simulations run fast.
2. Save it to `sumo/network/map.osm`.
3. Convert it to a SUMO network:
   ```bash
   netconvert --osm-files sumo/network/map.osm -o sumo/network/map.net.xml
   ```
4. (Optional but recommended) Use SUMO's `osmWebWizard.py` for a guided setup that also generates traffic demand:
   ```bash
   python "$SUMO_HOME/tools/osmWebWizard.py"
   ```
   This opens a browser map picker and generates network + routes + a ready `.sumocfg` in one step — a fast way to get your first working scenario before hand-crafting charging stations.
5. Add charging stations as an "additional file" (`sumo/stations/chargers.add.xml`) using SUMO's built-in `<chargingStation>` element, and reference EV-specific parameters (battery capacity, consumption) via the `<param>` elements on vehicles — SUMO's Battery/`sumo/tools/sumolib` route documents this under "Electric Vehicles" in the SUMO docs.

---

## 8. Sanity-check the TraCI connection

Save as `src/simulation/test_connection.py`:
```python
import os, sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set the SUMO_HOME environment variable")

import traci

sumo_cfg = "sumo/config/your_scenario.sumocfg"  # update once you have one
sumo_cmd = ["sumo", "-c", sumo_cfg]

traci.start(sumo_cmd)
for step in range(50):
    traci.simulationStep()
    print("Vehicles on road:", traci.vehicle.getIDList())
traci.close()
```
Run it:
```bash
python src/simulation/test_connection.py
```
If you see vehicle ID lists printing, Python ↔ SUMO communication is working.

---

## 9. SQLite (now) → PostgreSQL (later)

SQLite needs no setup — Python's `sqlite3` module is built in. A minimal starting point:
```python
import sqlite3
conn = sqlite3.connect("data/db.sqlite3")
conn.execute("""
CREATE TABLE IF NOT EXISTS charging_stations (
    id INTEGER PRIMARY KEY,
    name TEXT,
    lat REAL, lon REAL,
    num_ports INTEGER,
    cost_per_kwh REAL
)
""")
conn.commit()
```
When you're ready to migrate, `sqlalchemy` (already in requirements.txt) lets you swap
the connection string from `sqlite:///data/db.sqlite3` to a `postgresql://...` URL with
minimal code changes. Install PostgreSQL locally only when you reach that stage
(Windows: installer from postgresql.org; macOS: `brew install postgresql`; Linux: `sudo apt install postgresql`).

---

## 10. Quick RL stack sanity check

Save as `src/rl_env/test_rl_stack.py`:
```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")   # placeholder env just to confirm the stack works
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=1000)
print("Stable-Baselines3 + Gymnasium + PyTorch stack OK")
```
This confirms PPO/Gymnasium/PyTorch are wired correctly before you build your custom
`EVChargingEnv` (which will wrap the TraCI connection as the environment's `step()`).

---

## 11. Next build steps (mapped to your project modules)

| Module (from your doc) | Where it lives | What it needs from this env |
|---|---|---|
| Simulation module | `src/simulation/` | traci, sumolib, a `.sumocfg` |
| RL module | `src/rl_env/` + training script | gymnasium.Env subclass, stable-baselines3 PPO |
| Route planning | `src/route_planning/` | networkx graph built from `sumolib.net` |
| Charging station management | `src/station_management/` | traci `chargingstation` API + pandas/sqlite |
| Recommendation engine | `src/recommendation_engine/` | trained PPO policy + station_management outputs |

A natural build order: (1) get a working SUMO scenario with a couple of charging
stations via osmWebWizard, (2) get TraCI reading vehicle/station state into a pandas
DataFrame, (3) wrap that into a Gymnasium `EVChargingEnv`, (4) train PPO, (5) add the
nearest-station baseline for comparison per your evaluation section.

---

## Troubleshooting

- **`ModuleNotFoundError: traci`** → SUMO_HOME not set, or you didn't add `$SUMO_HOME/tools` to `sys.path` (see step 8).
- **`sumo: command not found`** → SUMO's `bin` folder isn't on PATH; re-check step 5.
- **PyTorch install issues on Windows/GPU** → if you have an NVIDIA GPU and want CUDA acceleration, install the matching build from https://pytorch.org/get-started/locally/ instead of the plain pip version in requirements.txt.
- **netconvert produces a huge/slow network** → pick a smaller bounding box in the OSM export; large city-wide maps make early testing painfully slow.
