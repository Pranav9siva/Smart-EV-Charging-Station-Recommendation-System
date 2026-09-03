import json
from pathlib import Path

from scripts.run_dashboard import apply_control_action


def test_apply_control_action_updates_runtime_control_file(tmp_path: Path) -> None:
    control_path = tmp_path / "simulation_control.json"

    pause_state = apply_control_action({"action": "pause"}, control_path=control_path)
    assert pause_state["status"] == "paused"

    speed_state = apply_control_action({"action": "set_speed", "speed": 2.0}, control_path=control_path)
    assert speed_state["speed"] == 2.0

    reset_state = apply_control_action({"action": "reset"}, control_path=control_path)
    assert reset_state["reset"] is True

    persisted = json.loads(control_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "play"
    assert persisted["speed"] == 2.0
    assert persisted["reset"] is True
