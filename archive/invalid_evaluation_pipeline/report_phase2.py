from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "recommendations.sqlite"
STATIONS_LOOKUP = ROOT / "data" / "stations" / "stations_lookup.json"
FLEET_MANIFEST = ROOT / "data" / "fleet" / "fleet_manifest.json"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rows() -> pd.DataFrame:
    with sqlite3.connect(DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_log (
                run_id TEXT NOT NULL,
                vehicle_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                soc_pct_at_decision REAL NOT NULL,
                station_id TEXT NOT NULL,
                score REAL NOT NULL,
                travel_time_min REAL NOT NULL,
                queue_wait_min REAL NOT NULL,
                charge_time_min REAL NOT NULL,
                chosen INTEGER NOT NULL
            )
            """
        )
        rows = conn.execute(
            "SELECT vehicle_id, soc_pct_at_decision, station_id, score, travel_time_min, queue_wait_min, charge_time_min FROM recommendation_log"
        ).fetchall()
    return pd.DataFrame(
        rows,
        columns=["vehicle_id", "soc_pct_at_decision", "station_id", "score", "travel_time_min", "queue_wait_min", "charge_time_min"],
    )


def main() -> None:
    rows = load_rows()
    if rows.empty:
        print("No recommendation logs found")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rows["vehicle_id"], rows["travel_time_min"], label="travel_time_min")
    ax.set_title("Recommendation travel time per EV")
    ax.set_ylabel("Minutes")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recommendation_times.png", dpi=150)

    stations_lookup = json.loads(STATIONS_LOOKUP.read_text(encoding="utf-8")) if STATIONS_LOOKUP.exists() else {}
    fleet_manifest = json.loads(FLEET_MANIFEST.read_text(encoding="utf-8")) if FLEET_MANIFEST.exists() else []

    points = []
    for vehicle in fleet_manifest:
        points.append({"vehicle_id": vehicle["vehicle_id"], "lat": 12.95, "lon": 77.62, "kind": "origin"})
    for station_id, station_data in stations_lookup.items():
        points.append({"vehicle_id": station_id, "lat": station_data.get("y", 0), "lon": station_data.get("x", 0), "kind": "station"})

    df = pd.DataFrame(points)
    fig_map = px.scatter_mapbox(df, lat="lat", lon="lon", color="kind", hover_name="vehicle_id", zoom=10, height=600)
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.write_html(OUT_DIR / "recommendation_map.html")

    print(f"Wrote {OUT_DIR / 'recommendation_times.png'}")
    print(f"Wrote {OUT_DIR / 'recommendation_map.html'}")


if __name__ == "__main__":
    main()
