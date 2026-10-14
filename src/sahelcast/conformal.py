"""Conformal calibration of quantile forecasts, one quantile at a time, with adaptive
conformal inference (ACI).

A forecast's 10 % quantile should have 10 % of observations below it, its median 50 %, its
90 % quantile 90 %. Models rarely manage this on their own: they are biased, or too
confident, or both. The fix used here is the asymmetric form of conformalized quantile
regression (Romano et al., 2019), applied to every quantile level tau:

    residuals  r = (y - q_tau) / s     over recent daylight hours; s = clear-sky capacity factor
    shift      d = the tau-quantile of r
    new q_tau  = q_tau + s * d         (so that a fraction tau of recent y fell below it)

Three refinements matter for solar power:
1. Asymmetric: each quantile moves on its own, so bias is corrected as well as width.
2. Normalized: residuals are divided by the clear-sky capacity factor, because an error of
   0.05 is huge at 7 a.m. and small at noon.
3. Mondrian (conditional): days are grouped by what the weather forecast expected (clear,
   mixed or cloudy, known at issue time), and each group is calibrated with its own past
   errors. Without this, intervals come out too wide on clear days and too narrow on cloudy
   days, which is right on average and wrong for decisions.

ACI then adapts the level used for each shift after every day:
    tau_eff <- tau_eff + gamma * (tau - observed share of y below the calibrated q_tau)
so calibration recovers quickly after a regime change such as a dust episode.

Timing is respected: a forecast issued on day d is only scored once its target day has been
observed, so its errors are used from day d + lag_days onwards.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from .dataset import qcol
from .metrics import DAYLIGHT_CF

MIN_RESIDUALS = 48  # fewer past daylight residuals than this: fall back to the pooled set
SCALE_FLOOR = 0.05
REGIME_BINS = [-np.inf, 0.6, 0.85, np.inf]
REGIME_LABELS = ["cloudy", "mixed", "clear"]


def forecast_regimes(fc: pd.DataFrame) -> pd.Series:
    """Label each (node, issue day) by the NWP forecast's daily clear-sky index.

    Uses only the weather forecast and the clear-sky model, so it is known at issue time.
    """
    g = fc.groupby(["node_id", "issue_time"])
    csi = g["cf_nwp"].transform("sum") / g["cf_clearsky"].transform("sum").replace(0, np.nan)
    return pd.cut(csi.fillna(1.0), REGIME_BINS, labels=REGIME_LABELS).astype(str)


def calibrate_series(
    g: pd.DataFrame,
    quantiles: list[float],
    window_days: int,
    lag_days: int,
    gamma: float,
    mondrian: bool = True,
) -> pd.DataFrame:
    """Calibrate one model's forecasts for one node. `g` has one row per target hour."""
    g = g.sort_values(["issue_time", "valid_time"]).copy()
    days = g["issue_time"].to_numpy()
    unique_days = np.unique(days)
    starts = np.searchsorted(days, unique_days, side="left")
    ends = np.searchsorted(days, unique_days, side="right")
    y = g["y"].to_numpy()
    scale = np.maximum(g["cf_clearsky"].to_numpy(), SCALE_FLOOR)
    daylight = (g["cf_clearsky"].to_numpy() > DAYLIGHT_CF) & ~np.isnan(y)
    labels = (
        g["regime_fc"].to_numpy() if (mondrian and "regime_fc" in g) else np.full(len(g), "all")
    )
    day_group = labels[starts]
    new = {}

    for tau in quantiles:
        raw = g[qcol(tau)].to_numpy()
        cal = raw.copy()
        tau_eff: dict[str, float] = {}
        pools: dict[str, deque[tuple[int, np.ndarray]]] = {}
        pending: deque[int] = deque()  # days whose outcomes are not known yet
        for i in range(len(unique_days)):
            # 1. Outcomes of forecasts issued lag_days ago or earlier are known now.
            while pending and pending[0] <= i - lag_days:
                j = pending.popleft()
                rows_j = slice(starts[j], ends[j])
                mask = daylight[rows_j]
                if mask.any():
                    yj = y[rows_j][mask]
                    k = day_group[j]
                    below = float(np.mean(yj < cal[rows_j][mask]))
                    tau_eff[k] = float(
                        np.clip(tau_eff.get(k, tau) + gamma * (tau - below), 1e-3, 1 - 1e-3)
                    )
                    resid = (yj - raw[rows_j][mask]) / scale[rows_j][mask]
                    pools.setdefault(k, deque()).append((j, resid))
                    pools.setdefault("__all__", deque()).append((j, resid))
            for pool in pools.values():
                while pool and pool[0][0] < i - lag_days - window_days + 1:
                    pool.popleft()
            # 2. Shift today's quantile using the recent errors of days like today.
            k = day_group[i]
            own = pools.get(k, deque())
            chosen = (
                own
                if sum(len(r) for _, r in own) >= MIN_RESIDUALS
                else pools.get("__all__", deque())
            )
            residuals = np.concatenate([r for _, r in chosen]) if chosen else np.array([])
            if len(residuals) >= MIN_RESIDUALS:
                rows = slice(starts[i], ends[i])
                cal[rows] = raw[rows] + scale[rows] * float(
                    np.quantile(residuals, tau_eff.get(k, tau))
                )
            pending.append(i)
        new[tau] = cal

    # Clip to what is physically possible (0 to the inverter limit) and keep quantiles in
    # order. Do not cap at the clear-sky value: real output can exceed a clear-sky model.
    matrix = np.column_stack([np.clip(new[q], 0.0, 1.0) for q in quantiles])
    matrix.sort(axis=1)
    matrix[g["cf_clearsky"].to_numpy() <= DAYLIGHT_CF] = 0.0
    for k, q in enumerate(quantiles):
        g[qcol(q)] = matrix[:, k]
    return g


def calibrate(
    fc: pd.DataFrame,
    quantiles: list[float],
    window_days: int = 60,
    lag_days: int = 2,
    gamma: float = 0.005,
    mondrian: bool = True,
) -> pd.DataFrame:
    """Calibrate every (model, node) forecast series. Model names get a '+cqr' suffix."""
    fc = fc.copy()
    if mondrian and "cf_nwp" in fc:
        fc["regime_fc"] = forecast_regimes(fc)
    parts = [
        calibrate_series(g, quantiles, window_days, lag_days, gamma, mondrian)
        for _, g in fc.groupby(["model", "node_id"], sort=False)
    ]
    out = pd.concat(parts, ignore_index=True)
    out["model"] = out["model"] + "+cqr"
    return out
