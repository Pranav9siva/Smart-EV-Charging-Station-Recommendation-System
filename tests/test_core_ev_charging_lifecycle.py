from pathlib import Path

import pytest

from scripts.evaluate_phase1_core_environment import run_phase1_core_environment

pytestmark = pytest.mark.skip(reason="Phase 1C deferred: run isolated Phase 1A and Phase 1B tests first.")


def test_phase1_core_environment_lifecycle() -> None:
    output = run_phase1_core_environment()
    output_dir = Path(output["output_dir"])

    assert (output_dir / "phase1_core_environment.json").exists()
    assert (output_dir / "phase1_events.csv").exists()
    assert (output_dir / "phase1_core_environment_summary.md").exists()

    assert output["complete_ev_lifecycle_passed"] is True

    tiny = next(item for item in output["scenarios"] if item["name"] == "tiny_1ev_2stations")
    medium = next(item for item in output["scenarios"] if item["name"] == "medium_10ev_20stations")

    assert tiny["checks"]["soc_increased"] is True
    assert tiny["checks"]["resumed_route"] is True
    assert tiny["checks"]["charging_completed"] is True

    assert medium["checks"]["multiple_independent_charging_or_queue"] is True
    assert medium["checks"]["charging_started"] is True
