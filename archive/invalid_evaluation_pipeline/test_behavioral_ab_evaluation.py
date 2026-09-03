from pathlib import Path

from scripts.behavioral_ab_evaluation import run_behavioral_ab_evaluation


def test_behavioral_ab_evaluation_writes_artifacts() -> None:
    output = run_behavioral_ab_evaluation()
    root = Path(output["output_dir"])

    assert (root / "behavioral_ab_results.csv").exists()
    assert (root / "behavioral_ab_results.json").exists()
    assert (root / "behavioral_ab_summary.md").exists()
    assert (root / "behavioral_ab_metrics.csv").exists()
    assert (root / "behavioral_ab_scenario_summary.csv").exists()
    assert output["results"]["original"]["status"] in {"PASSED", "FAILED_WITH_WARNINGS", "FAILED"}
    assert output["results"]["fine_tuned"]["status"] in {"PASSED", "FAILED_WITH_WARNINGS", "FAILED"}
