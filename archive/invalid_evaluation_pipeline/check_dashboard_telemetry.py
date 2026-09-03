from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "outputs" / "dashboard_state.json"


def main() -> int:
    if not STATE_PATH.exists():
        print("dashboard_state.json not found; run the simulation first.")
        return 1

    payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    simulation = payload.get("simulation") or {}
    ppo_decision = payload.get("ppo_decision") or {}
    history = payload.get("history") or {}
    tracked_ev = payload.get("tracked_ev") or {}

    print("simulation_step", simulation.get("step"))
    print("simulation_status", simulation.get("simulation_status"))
    print("tracked_ev_id", tracked_ev.get("id"))
    print("tracked_ev_station", tracked_ev.get("selected_station"))
    print("ppo_decision_ev", ppo_decision.get("ev_id"))
    print("ppo_decision_station", ppo_decision.get("selected_station"))
    print("ppo_decision_reason", ppo_decision.get("reason"))
    print("history_rewards", history.get("rewards", [])[:5])
    print("history_timestamps", history.get("timestamps", [])[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
