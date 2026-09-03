from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from pathlib import Path
from typing import Any

import pytest

TIMEOUT_SECONDS = 30


def _sumo_smoke_worker(config_path: str, steps: int, result_queue: mp.Queue) -> None:
    traci = None
    started = False
    try:
        from src.simulation.traci_interface import configure_traci_path, start_sumo

        configure_traci_path()
        import traci as traci_module

        traci = traci_module
        sumo_cmd = start_sumo(config_path, gui=False)
        traci.start(sumo_cmd)
        started = True

        connection_ok = True
        step_times = []
        for _ in range(max(1, min(5, int(steps)))):
            traci.simulationStep()
            step_times.append(float(traci.simulation.getTime()))

        result_queue.put(
            {
                "ok": True,
                "payload": {
                    "connected": connection_ok,
                    "steps_executed": len(step_times),
                    "step_times": step_times,
                    "connection_alive_after_steps": True,
                },
            }
        )
    except Exception:
        result_queue.put({"ok": False, "error": traceback.format_exc()})
    finally:
        if started and traci is not None:
            try:
                traci.close()
            except Exception:
                pass


def _run_sumo_smoke_with_timeout(config_path: Path, timeout_seconds: int = TIMEOUT_SECONDS) -> dict[str, Any]:
    if not config_path.exists():
        pytest.fail(f"SUMO config file not found: {config_path}")

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_sumo_smoke_worker, args=(str(config_path), 3, result_queue))

    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail(f"SUMO smoke test exceeded {timeout_seconds} seconds timeout.")

    if process.exitcode not in (0, None):
        pytest.fail(f"SUMO smoke worker exited with code {process.exitcode}.")

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        pytest.fail("SUMO smoke worker produced no result.")

    if not result.get("ok"):
        pytest.fail(f"SUMO/TraCI smoke test failed:\n{result.get('error', 'unknown error')}")

    return dict(result["payload"])


def test_sumo_connection_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "simulations" / "bangalore" / "sim.sumocfg"

    result = _run_sumo_smoke_with_timeout(config_path=config_path, timeout_seconds=TIMEOUT_SECONDS)

    assert result["connected"] is True
    assert result["steps_executed"] >= 1
    assert result["steps_executed"] <= 5
    assert result["connection_alive_after_steps"] is True
