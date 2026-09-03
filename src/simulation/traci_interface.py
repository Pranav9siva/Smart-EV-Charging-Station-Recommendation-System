import os
import sys


def ensure_sumo_home() -> str:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError("SUMO_HOME is not set. Please set SUMO_HOME to your SUMO installation folder.")
    return sumo_home


def configure_traci_path() -> None:
    sumo_home = ensure_sumo_home()
    tools_path = os.path.join(sumo_home, "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)


def start_sumo(config_path: str, gui: bool = False) -> list[str]:
    configure_traci_path()
    import traci

    sumo_binary = "sumo-gui" if gui else "sumo"
    return [sumo_binary, "-c", config_path]


def connect(config_path: str, gui: bool = False) -> "traci.Connection":
    configure_traci_path()
    import traci

    sumo_cmd = start_sumo(config_path, gui=gui)
    traci.start(sumo_cmd)
    return traci


def test_connection(config_path: str) -> bool:
    configure_traci_path()
    import traci

    sumo_cmd = start_sumo(config_path, gui=False)
    traci.start(sumo_cmd)
    traci.close()
    return True
