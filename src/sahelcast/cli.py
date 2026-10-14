"""Command-line interface. Run `sahelcast --help` (or `uv run sahelcast --help`)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .config import Settings, load_plants, load_settings

log = logging.getLogger("sahelcast")


def _date(text: str) -> date:
    return date.fromisoformat(text)


def best_model(lake, base: bool = True) -> str:
    """The calibrated model with the lowest national quantile score in the backtest."""
    m = lake.read("gold.metrics")
    m = m[(m["level"] == "country") & m["model"].str.endswith("+cqr")].sort_values("quantile_score")
    name = str(m["model"].iloc[0])
    return name.removesuffix("+cqr") if base else name


def cmd_ingest(s: Settings, args) -> None:
    from .ingest import ingest_history, make_client
    from .lake import Lake

    plants = load_plants(pv=s.pv)
    with Lake.from_settings(s) as lake, make_client(s) as client:
        counts = ingest_history(s, plants, lake, client, args.start, args.end, tuple(args.sources))
    log.info("rows loaded: %s", counts)


def cmd_ingest_live(s: Settings, args) -> None:
    from .ingest import ingest_live, make_client
    from .lake import Lake

    with Lake.from_settings(s) as lake, make_client(s) as client:
        n = ingest_live(s, load_plants(pv=s.pv), lake, client)
    log.info("live forecast rows: %s", n)


def cmd_transform(s: Settings, args) -> None:
    from .transform import run_dbt

    if not run_dbt(s, args.dbt_args or None):
        sys.exit("dbt failed: read the messages above")


def cmd_physics(s: Settings, args) -> None:
    from .lake import Lake
    from .physics import build_pv_hourly

    with Lake.from_settings(s) as lake:
        table = build_pv_hourly(lake, s)
    log.info("gold.pv_hourly: %s rows", len(table))


def cmd_backtest(s: Settings, args) -> None:
    from .backtest import run_backtest
    from .lake import Lake

    with Lake.from_settings(s) as lake:
        tables = run_backtest(lake, s, args.models, args.start, args.end)
    m = tables["metrics"]
    cols = ["model", "quantile_score", "coverage_80", "coverage_90", "skill_vs_persistence"]
    print(m[m["level"] == "country"][cols].sort_values("quantile_score").to_string(index=False))


def cmd_reconcile(s: Settings, args) -> None:
    from .lake import Lake
    from .reconcile import evaluate_reconciliation, reconcile_medians

    with Lake.from_settings(s) as lake:
        model = args.model or best_model(lake, base=False)
        fc = lake.read("gold.forecasts", f"WHERE model = '{model}'")
        rec = reconcile_medians(fc, lake.read("silver.stg_plants"))
        scores = evaluate_reconciliation(rec).assign(model=model)
        lake.replace("gold.reconciliation", rec.assign(model=model))
        lake.replace("gold.reconciliation_scores", scores)
    print(scores.to_string(index=False))


def cmd_dispatch(s: Settings, args) -> None:
    from .dispatch import run_dispatch_study, summarize_dispatch
    from .lake import Lake

    with Lake.from_settings(s) as lake:
        model = args.model or best_model(lake)
        fc = lake.read(
            "gold.forecasts", f"WHERE model IN ('{model}', '{model}+cqr') AND level = 'plant'"
        )
        results = run_dispatch_study(fc, model, s.battery, s.forecast.quantiles, args.days)
        summary = summarize_dispatch(results).assign(model=model)
        lake.replace("gold.dispatch_results", results)
        lake.replace("gold.dispatch_summary", summary)
        print(summary.to_string(index=False))
        if args.grid:
            rows = []
            for energy in (0.5, 1.0, 2.0):
                for penalty in (2.0, 3.0, 5.0):
                    b = s.battery.model_copy(
                        update={"energy_hours": energy, "shortfall_penalty": penalty}
                    )
                    r = summarize_dispatch(
                        run_dispatch_study(fc, model, b, s.forecast.quantiles, args.grid_days)
                    )
                    rows.append(r.assign(battery_hours=energy, penalty=penalty))
            grid = pd.concat(rows, ignore_index=True)
            grid = grid[
                ["battery_hours", "penalty", "policy", "regret_per_mw_day", "share_of_ideal_profit"]
            ]
            lake.replace("gold.dispatch_sensitivity", grid)
            print(grid.to_string(index=False))


def cmd_forecast(s: Settings, args) -> None:
    from .lake import Lake
    from .live import forecast_tomorrow

    with Lake.from_settings(s) as lake:
        fc = forecast_tomorrow(lake, s, args.models)
    national = fc[(fc["level"] == "country") & fc["model"].str.endswith("+cqr")]
    energy = (national["q50"] * national["ac_mw"]).groupby(national["model"]).sum()
    print("National median energy for tomorrow (MWh):")
    print(energy.round(0).to_string())


def cmd_publish(s: Settings, args) -> None:
    from .lake import Lake
    from .publish import publish

    with Lake.from_settings(s) as lake:
        written = publish(lake, s)
    log.info("serving tables: %s", ", ".join(written))


def cmd_report(s: Settings, args) -> None:
    from .report import build_site

    log.info("website written to %s", build_site(s))


def cmd_serve(s: Settings, args) -> None:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(s), host=args.host, port=args.port)


def cmd_snapshots(s: Settings, args) -> None:
    from .lake import Lake

    with Lake.from_settings(s, read_only=True) as lake:
        snaps = lake.snapshots()
    print(snaps[["snapshot_id", "snapshot_time", "changes"]].tail(args.last).to_string(index=False))


def cmd_pipeline(s: Settings, args) -> None:
    if not args.skip_ingest:
        cmd_ingest(
            s,
            argparse.Namespace(
                start=args.start, end=args.end, sources=["nwp", "satellite", "reanalysis"]
            ),
        )
    cmd_transform(s, argparse.Namespace(dbt_args=None))
    cmd_physics(s, args)
    cmd_backtest(
        s, argparse.Namespace(models=args.models, start=args.backtest_start, end=args.backtest_end)
    )
    cmd_reconcile(s, argparse.Namespace(model=None))
    cmd_dispatch(
        s, argparse.Namespace(model=None, days=args.dispatch_days, grid=False, grid_days=20)
    )
    cmd_publish(s, args)
    cmd_report(s, args)


def demo_settings(days: int = 240) -> Settings:
    """Offline demo: synthetic Open-Meteo, a separate data folder, a short recent window."""
    os.environ["SAHELCAST_OFFLINE"] = "1"
    s = load_settings()
    root = s.root / "data" / "demo"
    s.paths.lake_metadata = root / "lake" / "metadata.ducklake"
    s.paths.lake_data = root / "lake" / "files"
    s.paths.raw = root / "raw"
    s.paths.artifacts = root / "artifacts"
    s.paths.site = root / "site"
    end = date.today() - timedelta(days=7)
    s.openmeteo.history_start = end - timedelta(days=days - 1)
    s.openmeteo.history_end = end
    s.backtest.start = end - timedelta(days=int(days * 0.6))
    s.backtest.end = end - timedelta(days=1)
    return s


def cmd_demo(s: Settings, args) -> None:
    models = ["persistence", "physical", "lightgbm"] + (["chronos2"] if args.with_chronos else [])
    ns = argparse.Namespace(
        skip_ingest=False,
        start=None,
        end=None,
        models=models,
        backtest_start=None,
        backtest_end=None,
        dispatch_days=args.dispatch_days,
    )
    cmd_pipeline(s, ns)
    log.info("demo finished: open %s", Path(s.paths.site) / "index.html")


COMMANDS = {
    "ingest": cmd_ingest,
    "ingest-live": cmd_ingest_live,
    "transform": cmd_transform,
    "physics": cmd_physics,
    "backtest": cmd_backtest,
    "reconcile": cmd_reconcile,
    "dispatch": cmd_dispatch,
    "forecast": cmd_forecast,
    "publish": cmd_publish,
    "report": cmd_report,
    "serve": cmd_serve,
    "snapshots": cmd_snapshots,
    "pipeline": cmd_pipeline,
    "demo": cmd_demo,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sahelcast", description="Solar forecasting and battery dispatch for Algeria."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="download history from Open-Meteo into the bronze layer")
    p.add_argument("--start", type=_date)
    p.add_argument("--end", type=_date)
    p.add_argument(
        "--sources",
        nargs="+",
        default=["nwp", "satellite", "reanalysis"],
        choices=["nwp", "satellite", "reanalysis"],
    )
    sub.add_parser("ingest-live", help="download the latest forecast (run daily)")
    p = sub.add_parser("transform", help="build silver and gold with dbt (default: dbt build)")
    p.add_argument("dbt_args", nargs="*")
    sub.add_parser("physics", help="compute PV power with pvlib (gold.pv_hourly)")
    p = sub.add_parser("backtest", help="rolling-origin backtest, calibration and scores")
    p.add_argument(
        "--models", nargs="+", choices=["persistence", "physical", "lightgbm", "chronos2"]
    )
    p.add_argument("--start", type=_date)
    p.add_argument("--end", type=_date)
    p = sub.add_parser("reconcile", help="make plant, wilaya and national forecasts add up")
    p.add_argument("--model")
    p = sub.add_parser("dispatch", help="battery decisions and their value (regret)")
    p.add_argument("--model")
    p.add_argument("--days", type=int, default=60)
    p.add_argument(
        "--grid", action="store_true", help="also run the battery/penalty sensitivity grid"
    )
    p.add_argument("--grid-days", type=int, default=20)
    p = sub.add_parser("forecast", help="tomorrow's forecast (run ingest-live first)")
    p.add_argument(
        "--models", nargs="+", choices=["persistence", "physical", "lightgbm", "chronos2"]
    )
    sub.add_parser("publish", help="export serving tables for the API and the website")
    sub.add_parser("report", help="build the static website")
    p = sub.add_parser("serve", help="run the REST API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("snapshots", help="list the lake's snapshots (time travel)")
    p.add_argument("--last", type=int, default=20)
    p = sub.add_parser("pipeline", help="run every step in order")
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument("--start", type=_date)
    p.add_argument("--end", type=_date)
    p.add_argument(
        "--models", nargs="+", choices=["persistence", "physical", "lightgbm", "chronos2"]
    )
    p.add_argument("--backtest-start", type=_date)
    p.add_argument("--backtest-end", type=_date)
    p.add_argument("--dispatch-days", type=int, default=60)
    p = sub.add_parser("demo", help="the whole pipeline offline, on synthetic data (data/demo)")
    p.add_argument(
        "--with-chronos", action="store_true", help="also run Chronos-2 (downloads weights)"
    )
    p.add_argument("--dispatch-days", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = demo_settings() if args.command == "demo" else load_settings()
    COMMANDS[args.command](settings, args)


if __name__ == "__main__":
    main()
