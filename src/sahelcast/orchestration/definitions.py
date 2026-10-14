"""Dagster: the same pipeline steps as the CLI, as scheduled, observable assets.

Start the UI with `uv run dagster dev` (it reads [tool.dagster] in pyproject.toml), then open
http://127.0.0.1:3000. Two jobs:
- daily_forecast     every day at 12:15 UTC: fresh data, tomorrow's forecast, website
- weekly_evaluation  every Sunday: backtest, reconciliation, dispatch study, website

Everything runs in one process, one step at a time, because the lake's DuckDB catalog
accepts a single writer.
"""

from __future__ import annotations

from datetime import date, timedelta

import dagster as dg

from ..config import load_plants, load_settings
from ..lake import Lake


@dg.asset(
    group_name="ingest", description="Last 10 days of NWP, satellite and reanalysis data (bronze)."
)
def bronze_recent() -> dg.MaterializeResult:
    from ..ingest import ingest_history, make_client

    s = load_settings()
    with Lake.from_settings(s) as lake, make_client(s) as client:
        counts = ingest_history(
            s,
            load_plants(pv=s.pv),
            lake,
            client,
            start=date.today() - timedelta(days=10),
            end=date.today(),
        )
    return dg.MaterializeResult(metadata={k: dg.MetadataValue.int(v) for k, v in counts.items()})


@dg.asset(group_name="ingest", description="The latest Open-Meteo forecast run (bronze.nwp_live).")
def bronze_live() -> dg.MaterializeResult:
    from ..ingest import ingest_live, make_client

    s = load_settings()
    with Lake.from_settings(s) as lake, make_client(s) as client:
        n = ingest_live(s, load_plants(pv=s.pv), lake, client)
    return dg.MaterializeResult(metadata={"rows": dg.MetadataValue.int(n)})


@dg.asset(
    deps=[bronze_recent],
    group_name="transform",
    description="dbt build: silver and gold tables plus data tests.",
)
def dbt_models() -> None:
    from ..transform import run_dbt

    if not run_dbt(load_settings()):
        raise dg.Failure("dbt build failed")


@dg.asset(
    deps=[dbt_models], group_name="transform", description="PV power from pvlib (gold.pv_hourly)."
)
def pv_hourly() -> dg.MaterializeResult:
    from ..physics import build_pv_hourly

    s = load_settings()
    with Lake.from_settings(s) as lake:
        table = build_pv_hourly(lake, s)
    return dg.MaterializeResult(metadata={"rows": dg.MetadataValue.int(len(table))})


@dg.asset(
    deps=[pv_hourly, bronze_live],
    group_name="forecast",
    description="Tomorrow's calibrated forecast.",
)
def forecast_tomorrow() -> dg.MaterializeResult:
    from ..live import forecast_tomorrow as run

    s = load_settings()
    with Lake.from_settings(s) as lake:
        fc = run(lake, s, ["persistence", "physical", "lightgbm"])
    national = fc[(fc["node_id"] == "DZ") & (fc["model"] == "lightgbm+cqr")]
    energy = float((national["q50"] * national["ac_mw"]).sum())
    return dg.MaterializeResult(metadata={"national_median_mwh": dg.MetadataValue.float(energy)})


@dg.asset(
    deps=[pv_hourly], group_name="evaluate", description="Rolling-origin backtest with calibration."
)
def backtest() -> dg.MaterializeResult:
    from ..backtest import run_backtest

    s = load_settings()
    with Lake.from_settings(s) as lake:
        tables = run_backtest(lake, s)
    m = tables["metrics"]
    best = m[(m["level"] == "country")].sort_values("quantile_score").iloc[0]
    return dg.MaterializeResult(
        metadata={
            "best_model": dg.MetadataValue.text(str(best["model"])),
            "best_quantile_score": dg.MetadataValue.float(float(best["quantile_score"])),
            "coverage_80": dg.MetadataValue.float(float(best["coverage_80"])),
        }
    )


@dg.asset(
    deps=[backtest], group_name="evaluate", description="Reconciliation and battery dispatch study."
)
def decisions() -> None:
    from argparse import Namespace

    from ..cli import cmd_dispatch, cmd_reconcile

    s = load_settings()
    cmd_reconcile(s, Namespace(model=None))
    cmd_dispatch(s, Namespace(model=None, days=60, grid=False, grid_days=20))


@dg.asset(
    deps=[forecast_tomorrow],
    group_name="publish",
    description="Serving tables and the static website.",
)
def website() -> None:
    from ..publish import publish
    from ..report import build_site

    s = load_settings()
    with Lake.from_settings(s) as lake:
        publish(lake, s)
    build_site(s)


daily_job = dg.define_asset_job(
    "daily_forecast",
    selection=[bronze_recent, bronze_live, dbt_models, pv_hourly, forecast_tomorrow, website],
)
weekly_job = dg.define_asset_job("weekly_evaluation", selection=[backtest, decisions])

defs = dg.Definitions(
    assets=[
        bronze_recent,
        bronze_live,
        dbt_models,
        pv_hourly,
        forecast_tomorrow,
        backtest,
        decisions,
        website,
    ],
    jobs=[daily_job, weekly_job],
    schedules=[
        dg.ScheduleDefinition(job=daily_job, cron_schedule="15 12 * * *", execution_timezone="UTC"),
        dg.ScheduleDefinition(job=weekly_job, cron_schedule="0 3 * * 0", execution_timezone="UTC"),
    ],
    executor=dg.in_process_executor,
)
