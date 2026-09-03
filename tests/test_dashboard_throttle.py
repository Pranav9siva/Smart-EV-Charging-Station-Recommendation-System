import json

from src.dashboard.dashboard import Dashboard


def test_dashboard_research_refresh_is_throttled(tmp_path) -> None:
    dashboard = Dashboard(
        output_dir=str(tmp_path),
        fleet_size=2,
        station_count=1,
        tracked=1,
        dashboard_state_interval=1,
        dashboard_report_interval=3,
    )

    before = dashboard.research_dashboard_path.read_text(encoding="utf-8")

    writes = {"state": 0, "report": 0, "summary": 0}

    original_write_state = dashboard._write_state
    original_write_report = dashboard._write_research_dashboard
    original_write_summary = dashboard._write_summary

    def counted_write_state(state):
        writes["state"] += 1
        return original_write_state(state)

    def counted_write_report(state):
        writes["report"] += 1
        return original_write_report(state)

    def counted_write_summary(summary):
        writes["summary"] += 1
        return original_write_summary(summary)

    dashboard._write_state = counted_write_state  # type: ignore[method-assign]
    dashboard._write_research_dashboard = counted_write_report  # type: ignore[method-assign]
    dashboard._write_summary = counted_write_summary  # type: ignore[method-assign]

    state = {
        "vehicles": [{"id": "ev_1", "battery_pct": 40, "charging_status": "driving"}],
        "stations": [{"station_id": "st_1", "price_per_kwh": 0.2, "avg_wait_min": 1.0, "available_ports": 1, "grid_load_kw": 0.0}],
        "tracked_ids": ["ev_1"],
        "summary": {"charging_vehicles": 0, "avg_wait_min": 1.0},
        "metrics": [{"step": 1}],
    }

    dashboard.update(state)
    dashboard.update(state)

    after = dashboard.research_dashboard_path.read_text(encoding="utf-8")
    assert after == before
    assert writes["state"] == 2
    assert writes["report"] == 0
    assert writes["summary"] == 0

    dashboard.update(state)
    payload = json.loads(dashboard.state_path.read_text(encoding="utf-8"))
    assert payload["fleet_size"] == 2
    assert payload["tracked_ids"] == ["ev_1"]
    assert writes["state"] == 3
    assert writes["report"] == 1
    assert writes["summary"] == 1


def test_dashboard_finalize_forces_final_report(tmp_path) -> None:
    dashboard = Dashboard(
        output_dir=str(tmp_path),
        fleet_size=2,
        station_count=1,
        tracked=1,
        dashboard_state_interval=1,
        dashboard_report_interval=10,
    )

    writes = {"state": 0, "report": 0, "summary": 0}

    original_write_state = dashboard._write_state
    original_write_report = dashboard._write_research_dashboard
    original_write_summary = dashboard._write_summary

    dashboard._write_state = lambda state: writes.__setitem__("state", writes["state"] + 1) or original_write_state(state)  # type: ignore[method-assign]
    dashboard._write_research_dashboard = lambda state: writes.__setitem__("report", writes["report"] + 1) or original_write_report(state)  # type: ignore[method-assign]
    dashboard._write_summary = lambda summary: writes.__setitem__("summary", writes["summary"] + 1) or original_write_summary(summary)  # type: ignore[method-assign]

    dashboard.update({
        "vehicles": [{"id": "ev_1", "battery_pct": 40, "charging_status": "driving"}],
        "stations": [{"station_id": "st_1", "price_per_kwh": 0.2, "avg_wait_min": 1.0, "available_ports": 1, "grid_load_kw": 0.0}],
        "tracked_ids": ["ev_1"],
        "summary": {"charging_vehicles": 0, "avg_wait_min": 1.0},
        "metrics": [{"step": 1}],
    })
    dashboard.finalize()

    assert writes["state"] == 2
    assert writes["report"] == 1
    assert writes["summary"] == 1
