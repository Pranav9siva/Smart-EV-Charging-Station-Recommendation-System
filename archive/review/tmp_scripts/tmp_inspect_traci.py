import sys
sys.path.insert(0, '.')
import inspect
import traci

for name in ['changeTarget', 'setStop', 'setRoute', 'getRoute', 'setSpeed', 'getParameter', 'setParameter']:
    fn = getattr(traci.vehicle, name, None)
    print(name, 'exists' if fn else 'missing')
    if fn:
        try:
            print(inspect.signature(fn))
        except Exception as exc:
            print('signature error', exc)
    print('---')
