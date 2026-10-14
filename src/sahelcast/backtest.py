"""Rolling-origin backtest: re-create every past day's forecast as if it were issued that day.

For each issue day the models only see data available at 12:00 UTC that day (see
dataset.py). Then every forecast is calibrated with conformal prediction and scored. The
first `window_days + lag_days` days are a warm-up for the calibration and are not scored.

Outputs (written to the lake):
- gold.forecasts           every raw and calibrated forecast, with the observed value y
- gold.metrics             scores by model and level (plant / wilaya / country)
- gold.metrics_by_regime   the same, split by clear, mixed and overcast-or-dusty days
- gold.significance        Diebold-Mariano tests of each model against persistence
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from .baselines import forecast_lightgbm, forecast_persistence, forecast_physical
from .config import Settings
from .conformal import calibrate
from .dataset import Issue, Panel, build_panel, make_issues
from .lake import Lake
from .metrics import add_losses, add_skill, diebold_mariano, summarize
from .physics import daily_regimes

log = logging.getLogger(__name__)


def load_panel(lake: Lake, settings: Settings) -> pd.DataFrame:
    return build_panel(
        lake.read("gold.pv_hourly"), lake.read("silver.stg_plants"), settings.forecast.lead_day
    )


def run_models(
    panel: Panel, issues: list[Issue], settings: Settings, models: list[str], chronos=None
) -> pd.DataFrame:
    frames = []
    for name in models:
        log.info("forecasting with %s for %s issue days", name, len(issues))
        if name == "persistence":
            frames.append(forecast_persistence(panel, issues, settings))
        elif name == "physical":
            frames.append(forecast_physical(panel, issues, settings))
        elif name == "lightgbm":
            frames.append(forecast_lightgbm(panel, issues, settings))
        elif name == "chronos2":
            if chronos is None:
                try:
                    import chronos as _  # noqa: F401
                except ImportError:
                    log.warning(
                        "chronos-forecasting is not installed: skipping chronos2 (run `uv sync --all-extras`)"
                    )
                    continue
                from .chronos_model import Chronos2Forecaster

                chronos = Chronos2Forecaster(
                    settings.forecast.chronos_model_id, settings.forecast.device
                )
            frames.append(chronos.forecast(panel, issues, settings))
        else:
            raise ValueError(f"unknown model: {name}")
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def attach_truth(fc: pd.DataFrame, panel_df: pd.DataFrame, lead: int = 2) -> pd.DataFrame:
    """Add the observation (y) and the issue-time context each forecast row needs later."""
    cols = ["node_id", "valid_time", "level", "ac_mw", "cf_clearsky", f"cf_nwp{lead}", "target_cf"]
    out = fc.merge(panel_df[cols], on=["node_id", "valid_time"], how="left")
    return out.rename(columns={"target_cf": "y", f"cf_nwp{lead}": "cf_nwp"})


def evaluate(
    fc: pd.DataFrame, settings: Settings, panel_df: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    losses = add_losses(fc, settings.forecast.quantiles)
    losses["day"] = (losses["valid_time"] - pd.Timedelta(minutes=30)).dt.floor("D")
    regimes = daily_regimes(panel_df, key="node_id")
    losses = losses.merge(regimes, on=["node_id", "day"], how="left")

    overall = add_skill(summarize(losses, ["model", "level"]), by=["level"])
    by_regime = add_skill(summarize(losses, ["model", "level", "regime"]), by=["level", "regime"])

    daily = losses.groupby(["model", "level", "day"])["quantile_score"].mean()
    tests = []
    for (model, level), series in daily.groupby(level=["model", "level"]):
        if model.startswith("persistence"):
            continue
        ref = "persistence+cqr" if model.endswith("+cqr") else "persistence"
        if (ref, level) not in daily.index.droplevel("day"):
            continue
        a = series.droplevel(["model", "level"])
        b = daily.loc[(ref, level)]
        stat, p = diebold_mariano(a, b)
        tests.append(
            {"model": model, "reference": ref, "level": level, "dm_stat": stat, "p_value": p}
        )
    return {"metrics": overall, "metrics_by_regime": by_regime, "significance": pd.DataFrame(tests)}


def run_backtest(
    lake: Lake,
    settings: Settings,
    models: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    chronos=None,
    panel_hook=None,
) -> dict[str, pd.DataFrame]:
    """`panel_hook(panel_df) -> panel_df` can replace the target, e.g. with metered data."""
    bt, cal, fs = settings.backtest, settings.calibration, settings.forecast
    start, end = start or bt.start, end or bt.end
    warmup_days = cal.window_days + cal.lag_days
    if (end - start).days + 1 <= warmup_days:
        raise ValueError(
            f"a backtest needs more than {warmup_days} issue days: the first {warmup_days} only "
            "warm up the calibration and are not scored (to try a model quickly, see "
            "scripts/try_chronos.py)"
        )
    panel_df = load_panel(lake, settings)
    if panel_hook is not None:
        panel_df = panel_hook(panel_df)
    panel = Panel(panel_df)
    issues = make_issues(start, end, fs)
    raw = attach_truth(
        run_models(panel, issues, settings, models or fs.models, chronos), panel_df, fs.lead_day
    )
    calibrated = calibrate(
        raw, fs.quantiles, cal.window_days, cal.lag_days, cal.gamma, cal.mondrian
    )
    fc = pd.concat([raw, calibrated], ignore_index=True)
    warmup_end = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=cal.window_days + cal.lag_days)
    fc["in_evaluation"] = fc["issue_time"] >= warmup_end
    tables = evaluate(fc[fc["in_evaluation"]], settings, panel_df)
    lake.replace("gold.forecasts", fc)
    for name, table in tables.items():
        lake.replace(f"gold.{name}", table)
    tables["forecasts"] = fc
    return tables
