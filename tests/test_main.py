from __future__ import annotations

import main as root_main


def test_root_main_invokes_simulation_controller(monkeypatch, tmp_path) -> None:
    class DummyController:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def run(self, steps=None) -> None:
            self.steps = steps

        def stop(self) -> None:
            return None

    created: dict[str, object] = {}

    def fake_controller(*args, **kwargs):
        created["args"] = args
        created["kwargs"] = kwargs
        return DummyController(*args, **kwargs)

    monkeypatch.setattr(root_main, "SimulationController", fake_controller)

    sumo_cfg = tmp_path / "sim.sumocfg"
    sumo_cfg.write_text("<configuration />", encoding="utf-8")
    db_path = tmp_path / "stations.sqlite"
    db_path.write_text("db", encoding="utf-8")
    model_path = tmp_path / "model.zip"
    model_path.write_text("model", encoding="utf-8")

    exit_code = root_main.main([
        "--steps",
        "2",
        "--fleet-size",
        "3",
        "--tracked",
        "2",
        "--sumo-cfg",
        str(sumo_cfg),
        "--db",
        str(db_path),
        "--model",
        str(model_path),
    ])

    assert exit_code == 0
    assert created["kwargs"]["fleet_size"] == 3
    assert created["kwargs"]["tracked"] == 2
    assert created["kwargs"]["use_gui"] is False
