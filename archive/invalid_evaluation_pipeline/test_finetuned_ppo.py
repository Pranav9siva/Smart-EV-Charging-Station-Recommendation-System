from pathlib import Path

from scripts.legacy import test_finetuned_ppo as _implementation

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "stations" / "stations.sqlite"
make_env = _implementation.make_env
_implementation.DB_PATH = DB_PATH