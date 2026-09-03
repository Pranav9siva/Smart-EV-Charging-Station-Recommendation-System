from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import traci
from sumolib import checkBinary
from sumolib.miscutils import getFreeSocketPort

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'simulations' / 'bangalore' / 'sim.sumocfg'


def resolve_sumo_binary() -> str:
    for candidate in ('sumo', 'sumo-gui', 'sumo.exe', 'sumo-gui.exe'):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    try:
        return str(checkBinary('sumo'))
    except Exception as exc:
        return f'<unresolved: {exc}>'


def load_config_paths(cfg_path: Path) -> dict[str, Any]:
    import xml.etree.ElementTree as ET
    root = ET.parse(cfg_path).getroot()
    input_node = root.find('input')
    if input_node is None:
        raise RuntimeError('No <input> section found in SUMO config')
    net_file = input_node.find('net-file')
    route_files = input_node.find('route-files')
    additional_files = input_node.find('additional-files')
    net_value = net_file.attrib.get('value') if net_file is not None else None
    route_value = route_files.attrib.get('value') if route_files is not None else None
    add_value = additional_files.attrib.get('value') if additional_files is not None else None
    return {
        'net': net_value,
        'routes': route_value,
        'additional': add_value,
    }


def validate_path(label: str, value: str | None, cfg_dir: Path) -> str | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (cfg_dir / path).resolve()
    return str(path) if path.exists() else f'<missing: {path}>'


def stop_sumo(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    try:
        traci.close()
    except Exception:
        pass


def main() -> int:
    print('SUMO_BINARY:')
    binary = resolve_sumo_binary()
    print(binary)

    if not CONFIG.exists():
        print('FAILURE: SUMO config does not exist')
        return 1

    cfg_dir = CONFIG.parent
    try:
        resolved_paths = load_config_paths(CONFIG)
    except Exception as exc:
        print(f'FAILURE: Unable to parse SUMO config: {exc}')
        return 1

    network = validate_path('network', resolved_paths.get('net'), cfg_dir)
    routes = validate_path('routes', resolved_paths.get('routes'), cfg_dir)
    additional = validate_path('additional', resolved_paths.get('additional'), cfg_dir)

    print('SUMO_CONFIG:')
    print(str(CONFIG))
    print('NETWORK:')
    print(network if network is not None else '<none>')
    print('ROUTES:')
    print(routes if routes is not None else '<none>')
    print('ADDITIONAL_FILES:')
    print(additional if additional is not None else '<none>')

    if network and str(network).startswith('<missing'):
        print('FAILURE: network file missing')
        return 1
    if routes and str(routes).startswith('<missing'):
        print('FAILURE: route file missing')
        return 1
    if additional and str(additional).startswith('<missing'):
        print('FAILURE: additional file missing')
        return 1

    proc: subprocess.Popen[Any] | None = None
    stderr_text = ''
    port = int(getFreeSocketPort() or 8813)
    try:
        cmd = [binary, '-c', str(CONFIG), '--remote-port', str(port), '--no-step-log', 'true', '--verbose', 'false', '--time-to-teleport', '-1']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(2)
        print('SUMO_STARTED:')
        print('yes' if proc.poll() is None else 'no')
        try:
            traci.init(port=port, numRetries=1, host='127.0.0.1')
            print('TRACI_CONNECTED:')
            print('yes')
            print('FAILURE:')
            print('<none>')
        except Exception as exc:
            print('TRACI_CONNECTED:')
            print('no')
            print('FAILURE:')
            print(repr(exc))
        try:
            stdout_text, _ = proc.communicate(timeout=2)
            stderr_text = stdout_text or ''
        except subprocess.TimeoutExpired:
            proc.terminate()
            stdout_text, _ = proc.communicate(timeout=5)
            stderr_text = stdout_text or ''
        if stderr_text:
            print('SUMO_STDERR:')
            print(stderr_text)
    finally:
        stop_sumo(proc)

    return 0


if __name__ == '__main__':
    sys.exit(main())
