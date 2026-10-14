"""Three forecasting models that every serious solar forecast must beat.

1. persistence  "tomorrow looks like the last two weeks": the clear-sky index observed at
                each hour over the last 14 days, applied to tomorrow's clear-sky power.
2. physical     the weather model's irradiance run through the PV model (pvlib), with
                uncertainty taken from its own recent errors.
3. lightgbm     gradient boosting that learns the weather model's errors and adds them to
                the physical forecast, one model per quantile, retrained every month on
                everything observed before.

Each function returns one row per (node, target hour) with columns q05 ... q95 holding
capacity factors (0 to 1). Every function only reads data available at issue time.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Settings
from .dataset import Issue, Panel, qcol

log = logging.getLogger(__name__)

NIGHT_CF = 1e-3


def finalize(values: np.ndarray, cf_clearsky: np.ndarray) -> np.ndarray:
    """Clip to [0, 1], put the quantiles in order and set night hours to zero."""
    values = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
    values.sort(axis=1)
    values[cf_clearsky <= NIGHT_CF] = 0.0
    return values


def _frame(
    model: str,
    node: str,
    issue: Issue,
    target: pd.DataFrame,
    values: np.ndarray,
    quantiles: list[float],
) -> pd.DataFrame:
    out = pd.DataFrame(values, columns=[qcol(q) for q in quantiles])
    out.insert(0, "valid_time", target.index)
    out.insert(0, "issue_time", issue.time)
    out.insert(0, "node_id", node)
    out.insert(0, "model", model)
    return out


def forecast_persistence(
    panel: Panel, issues: list[Issue], settings: Settings, days: int = 14
) -> pd.DataFrame:
    quantiles = settings.forecast.quantiles
    out = []
    for node in panel.nodes:
        for issue in issues:
            hist = panel.history(node, issue.cutoff, days)
            target = panel.window(node, issue.target_start, issue.target_end)
            if target.empty:
                continue
            ok = hist["cf_clearsky"] > 0.05
            csi = (hist.loc[ok, "target_cf"] / hist.loc[ok, "cf_clearsky"]).clip(0, 1.2).dropna()
            hours = (csi.index - pd.Timedelta(minutes=30)).hour
            by_hour = {h: v.to_numpy() for h, v in csi.groupby(hours)}
            pooled = csi.to_numpy() if len(csi) else np.array([1.0])
            t_hours = (target.index - pd.Timedelta(minutes=30)).hour
            rows = []
            for h in t_hours:
                sample = by_hour.get(h)
                sample = sample if sample is not None and len(sample) >= 3 else pooled
                rows.append(np.quantile(sample, quantiles))
            cs = target["cf_clearsky"].to_numpy()
            values = np.asarray(rows) * cs[:, None]
            out.append(_frame("persistence", node, issue, target, finalize(values, cs), quantiles))
    return pd.concat(out, ignore_index=True)


def forecast_physical(
    panel: Panel, issues: list[Issue], settings: Settings, days: int = 60
) -> pd.DataFrame:
    quantiles = settings.forecast.quantiles
    lead = settings.forecast.lead_day
    point_col = f"cf_nwp{lead}"
    out = []
    for node in panel.nodes:
        for issue in issues:
            target = panel.window(node, issue.target_start, issue.target_end)
            if target.empty:
                continue
            hist = panel.history(node, issue.cutoff, days)
            ok = (hist["cf_clearsky"] > 0.05) & hist["target_cf"].notna() & hist[point_col].notna()
            resid = (
                (hist.loc[ok, "target_cf"] - hist.loc[ok, point_col]) / hist.loc[ok, "cf_clearsky"]
            ).to_numpy()
            offsets = (
                np.quantile(resid, quantiles) if len(resid) >= 24 else np.zeros(len(quantiles))
            )
            cs = target["cf_clearsky"].to_numpy()
            point = target[point_col].fillna(target["cf_clearsky"]).to_numpy()
            values = point[:, None] + cs[:, None] * offsets[None, :]
            out.append(_frame("physical", node, issue, target, finalize(values, cs), quantiles))
    return pd.concat(out, ignore_index=True)


def lightgbm_features(frame: pd.DataFrame, lead: int) -> pd.DataFrame:
    """Issue-time information only, in units that do not change with the seasons.

    Temperature and day-of-year are left out on purpose: pvlib already models their effect,
    and trees cannot extrapolate, so a model trained on winter would be lost in its first
    summer.
    """
    t = frame.index - pd.Timedelta(minutes=30)
    clear = frame["cf_clearsky"].where(frame["cf_clearsky"] > 0.01)
    return pd.DataFrame(
        {
            f"csi_nwp{lead}": frame[f"cf_nwp{lead}"] / clear,
            f"nwp{lead}_cloud_cover": frame[f"nwp{lead}_cloud_cover"],
            "cf_clearsky": frame["cf_clearsky"],
            "solar_zenith": frame["solar_zenith"],
            "hour_sin": np.sin(2 * np.pi * t.hour / 24),
            "hour_cos": np.cos(2 * np.pi * t.hour / 24),
            "latitude": frame["latitude"],
        },
        index=frame.index,
    )


def forecast_lightgbm(
    panel: Panel, issues: list[Issue], settings: Settings, max_rows: int = 200_000
) -> pd.DataFrame:
    """Learn the weather model's errors, then add them to the physical forecast.

    The target is the normalized residual r = (y - physical) / clear-sky capacity factor.
    One LightGBM model per quantile predicts r; the forecast is physical + clear-sky x r.
    Physics handles what physics knows (sun, temperature, the PV system); the model learns
    only where the weather forecast goes wrong.
    """
    quantiles = settings.forecast.quantiles
    lead = settings.forecast.lead_day
    point_col = f"cf_nwp{lead}"
    every = settings.backtest.retrain_every_days
    df = panel.df.set_index("valid_time")
    out = []
    for start in range(0, len(issues), every):
        block = issues[start : start + every]
        cutoff = block[0].cutoff
        train = df[
            (df.index <= cutoff)
            & (df["cf_clearsky"] > 0.05)
            & df["target_cf"].notna()
            & df[point_col].notna()
        ]
        if len(train) < 500:
            log.warning(
                "lightgbm: only %s training rows before %s, skipping block", len(train), cutoff
            )
            continue
        if len(train) > max_rows:
            train = train.sample(max_rows, random_state=0)
        x_train = lightgbm_features(train, lead)
        residual = (train["target_cf"] - train[point_col]) / train["cf_clearsky"]
        models = {}
        for q in quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=100,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.9,
                verbose=-1,
            )
            model.fit(x_train, residual)
            models[q] = model
        for node in panel.nodes:
            for issue in block:
                target = panel.window(node, issue.target_start, issue.target_end)
                if target.empty:
                    continue
                x = lightgbm_features(target, lead)
                cs = target["cf_clearsky"].to_numpy()
                point = target[point_col].fillna(target["cf_clearsky"]).to_numpy()
                r = np.column_stack([models[q].predict(x) for q in quantiles])
                values = point[:, None] + cs[:, None] * r
                out.append(_frame("lightgbm", node, issue, target, finalize(values, cs), quantiles))
        log.info("lightgbm: block starting %s done", block[0].day)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
