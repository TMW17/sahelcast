"""Try Chronos-2 on one week before running it in a full backtest (Phase 3, step 3.5).

    uv run python scripts/try_chronos.py 2025-06-01

It reads gold.pv_hourly (run `sahelcast physics` first), forecasts seven issue days with the
model in settings.yaml, prints how long that took and the national forecast for the first
day. Nothing is written to the lake. The first run downloads the model from Hugging Face.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from time import perf_counter

from sahelcast.backtest import load_panel
from sahelcast.chronos_model import Chronos2Forecaster
from sahelcast.config import Settings, load_settings
from sahelcast.dataset import Panel, make_issues
from sahelcast.lake import Lake


def main(start: date, settings: Settings | None = None, days: int = 7, pipeline=None):
    s = settings or load_settings()
    with Lake.from_settings(s, read_only=True) as lake:
        panel = Panel(load_panel(lake, s))
    issues = make_issues(start, start + timedelta(days=days - 1), s.forecast)
    model = Chronos2Forecaster(s.forecast.chronos_model_id, s.forecast.device, pipeline)
    t0 = perf_counter()
    fc = model.forecast(panel, issues, s)
    seconds = perf_counter() - t0
    print(f"{seconds:.0f} s for {days} issue days x {len(panel.nodes)} nodes")
    print(f"a full year would take about {seconds / days * 365 / 60:.0f} minutes")
    national = fc[(fc["node_id"] == "DZ") & (fc["issue_time"] == issues[0].time)]
    print(f"national capacity factor for {issues[0].target_start.date()}:")
    print(national.set_index("valid_time").filter(regex=r"^q\d+$").round(3).to_string())
    return fc


if __name__ == "__main__":
    main(date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2025, 6, 1))
