"""Benchmark SahelCast's method on metered Belgian solar data (Elia).

Run on your own machine (it downloads about 2 years of Open-Meteo and Elia data):
    uv run python scripts/benchmark_elia.py

What it does:
1. builds a separate lake in data/belgium for six representative Belgian points
2. downloads Elia's national measured PV and its day-ahead forecast bounds
3. runs the same backtest with Elia's metered output as the national observation
4. prints SahelCast's scores next to Elia's own forecast scores (P10, P50, P90)

This script was written against Elia's documented export API but not run against it while
the starter was built: check the field names first (see src/sahelcast/elia.py).
"""

from __future__ import annotations

import os
from argparse import Namespace
from datetime import date, timedelta

os.environ.setdefault("SAHELCAST_PLANTS", "config/plants_belgium.csv")

from sahelcast import cli  # noqa: E402
from sahelcast.backtest import run_backtest  # noqa: E402
from sahelcast.config import load_settings  # noqa: E402
from sahelcast.elia import (  # noqa: E402
    download_elia,
    elia_scores,
    our_scores,
    replace_target,
    to_hourly,
)
from sahelcast.lake import Lake  # noqa: E402

HISTORY_START, HISTORY_END = date(2024, 3, 1), date(2025, 12, 31)
BACKTEST_START, BACKTEST_END = date(2025, 1, 1), date(2025, 12, 30)


def main() -> None:
    s = load_settings()
    base = s.root / "data" / "belgium"
    s.paths.lake_metadata = base / "lake" / "metadata.ducklake"
    s.paths.lake_data = base / "lake" / "files"
    s.paths.raw = base / "raw"
    s.paths.artifacts = base / "artifacts"
    s.paths.site = base / "site"
    s.backtest.start, s.backtest.end = BACKTEST_START, BACKTEST_END

    cli.cmd_ingest(
        s,
        Namespace(start=HISTORY_START, end=HISTORY_END, sources=["nwp", "satellite", "reanalysis"]),
    )
    cli.cmd_transform(s, Namespace(dbt_args=None))
    cli.cmd_physics(s, Namespace())

    elia = to_hourly(download_elia(HISTORY_START, HISTORY_END + timedelta(days=1)))
    with Lake.from_settings(s) as lake:
        tables = run_backtest(
            lake,
            s,
            models=["persistence", "physical", "lightgbm"],
            panel_hook=lambda panel: replace_target(panel, elia, node="BE"),
        )
    fc = tables["forecasts"]
    fc = fc[fc["in_evaluation"] & (fc["node_id"] == "BE")]
    period = elia[
        (elia["valid_time"] >= fc["valid_time"].min())
        & (elia["valid_time"] <= fc["valid_time"].max())
    ]
    print("Elia day-ahead (11h) forecast:", elia_scores(period))
    for model in sorted(fc["model"].unique()):
        print(f"SahelCast {model}:", our_scores(fc[fc["model"] == model]))


if __name__ == "__main__":
    main()
