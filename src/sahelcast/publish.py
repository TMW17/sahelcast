"""Export small, read-only "serving" tables for the API and the website.

The API and the website never open the lake. They read Parquet files in
data/artifacts/serving, so they keep working while the pipeline writes to the lake (a
DuckDB-file catalog allows only one process at a time).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Settings
from .lake import Lake

TABLES = {
    "forecasts": "gold.forecasts",
    "forecasts_live": "gold.forecasts_live",
    "metrics": "gold.metrics",
    "metrics_by_regime": "gold.metrics_by_regime",
    "significance": "gold.significance",
    "reconciliation": "gold.reconciliation_scores",
    "dispatch_summary": "gold.dispatch_summary",
    "dispatch_sensitivity": "gold.dispatch_sensitivity",
    "plants": "silver.stg_plants",
}


def serving_dir(settings: Settings) -> Path:
    return Path(settings.paths.artifacts) / "serving"


def publish(lake: Lake, settings: Settings, keep_days: int = 60) -> list[str]:
    out = serving_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in TABLES.items():
        if not lake.table_exists(table):
            continue
        frame = lake.read(table)
        if name == "forecasts" and not frame.empty:
            last = frame["issue_time"].max()
            frame = frame[frame["issue_time"] > last - pd.Timedelta(days=keep_days)]
        frame.to_parquet(out / f"{name}.parquet", index=False)
        written.append(name)
    (out / "SYNTHETIC").unlink(missing_ok=True)
    if settings.offline:
        (out / "SYNTHETIC").write_text("These tables were computed on synthetic demo data.\n")
    return written


def read_serving(settings: Settings, name: str) -> pd.DataFrame:
    path = serving_dir(settings) / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()
