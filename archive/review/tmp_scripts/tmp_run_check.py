from pathlib import Path
import sys
print(sys.version)
print(sys.executable)
print(Path.cwd())
try:
    import sumolib
    print('sumolib OK')
except Exception as exc:
    print('sumolib FAIL', exc)
try:
    import traci
    print('traci OK')
except Exception as exc:
    print('traci FAIL', exc)
try:
    import gymnasium
    print('gymnasium OK')
except Exception as exc:
    print('gymnasium FAIL', exc)
