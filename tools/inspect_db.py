from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "recommendations.sqlite"
conn = sqlite3.connect(DB_PATH)
print(conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
print(conn.execute("SELECT COUNT(*) FROM recommendation_log").fetchone())
