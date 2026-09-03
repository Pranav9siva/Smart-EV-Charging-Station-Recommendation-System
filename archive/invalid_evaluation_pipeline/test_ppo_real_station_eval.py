from pathlib import Path

from scripts.test_finetuned_ppo import DB_PATH, make_env


def test_real_station_environment_uses_sqlite_database() -> None:
    env = make_env()
    try:
        obs, _ = env.reset(seed=7)
        assert isinstance(obs, dict)
        assert "vehicles" in obs and "stations" in obs
        assert Path(DB_PATH).exists(), "expected SQLITE station database to exist"
        assert env.station_manager is not None
        assert len(env.station_manager.list_all_stations()) > 0
        assert any(candidate.get("station_id") for candidate in env.current_candidates)
    finally:
        env.close()
