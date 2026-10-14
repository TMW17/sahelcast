"""Scores for probabilistic forecasts.

All scores use daylight hours only (clear-sky capacity factor above 1 %). At night every
model is trivially right, which would make everyone look better than they are.

- pinball loss: the proper score for one quantile (lower is better)
- quantile score: mean pinball over all quantiles; 2 x this approximates the CRPS
- coverage: how often the truth falls inside a central interval (should match its level)
- width: average interval width (narrower is better, at correct coverage)
- skill: 1 - score / score of the persistence reference
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .dataset import qcol

DAYLIGHT_CF = 0.01


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    diff = y - q
    return np.maximum(tau * diff, (tau - 1) * diff)


def interval_pairs(quantiles: list[float]) -> list[tuple[float, float, float]]:
    """(lower quantile, upper quantile, nominal level) for each symmetric pair."""
    lows = [q for q in quantiles if q < 0.5]
    return [
        (lo, round(1 - lo, 4), round(1 - 2 * lo, 4)) for lo in lows if round(1 - lo, 4) in quantiles
    ]


def add_losses(fc: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    """Row-level losses and hits, for the daylight rows that have an observation."""
    fc = fc[(fc["cf_clearsky"] > DAYLIGHT_CF) & fc["y"].notna()].copy()
    y = fc["y"].to_numpy()
    losses = [pinball(y, fc[qcol(q)].to_numpy(), q) for q in quantiles]
    fc["quantile_score"] = np.mean(losses, axis=0)
    fc["abs_error"] = np.abs(y - fc["q50"].to_numpy())
    fc["sq_error"] = (y - fc["q50"].to_numpy()) ** 2
    for lo, hi, level in interval_pairs(quantiles):
        tag = f"{round(level * 100)}"
        inside = (y >= fc[qcol(lo)].to_numpy()) & (y <= fc[qcol(hi)].to_numpy())
        fc[f"hit_{tag}"] = inside.astype(float)
        fc[f"width_{tag}"] = fc[qcol(hi)].to_numpy() - fc[qcol(lo)].to_numpy()
    return fc


def summarize(losses: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    hit_cols = [c for c in losses.columns if c.startswith(("hit_", "width_"))]
    agg = losses.groupby(by, observed=True).agg(
        n_hours=("y", "size"),
        quantile_score=("quantile_score", "mean"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda s: float(np.sqrt(s.mean()))),
        **{c: (c, "mean") for c in hit_cols},
    )
    agg["crps_approx"] = 2 * agg["quantile_score"]
    return agg.rename(columns=lambda c: c.replace("hit_", "coverage_")).reset_index()


def add_skill(
    summary: pd.DataFrame, reference: str = "persistence", by: list[str] | None = None
) -> pd.DataFrame:
    """Skill of every model relative to the reference model, within the same group."""
    by = by or []
    ref = summary[summary["model"] == reference].set_index(by)["quantile_score"] if by else None
    if by:
        denom = summary.set_index(by).index.map(ref)
    else:
        denom = summary.loc[summary["model"] == reference, "quantile_score"].iloc[0]
    summary["skill_vs_" + reference] = 1 - summary["quantile_score"] / np.asarray(
        denom, dtype=float
    )
    return summary


def diebold_mariano(daily_loss_a: pd.Series, daily_loss_b: pd.Series) -> tuple[float, float]:
    """Is model A's daily loss significantly different from model B's?

    Returns (statistic, two-sided p-value). Negative statistic = A is better. Uses a
    Newey-West variance with 7 lags to allow for day-to-day correlation of errors.
    """
    d = (daily_loss_a - daily_loss_b).dropna().to_numpy()
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    d_c = d - d.mean()
    var = np.sum(d_c**2) / n
    for lag in range(1, 8):
        w = 1 - lag / 8
        var += 2 * w * np.sum(d_c[lag:] * d_c[:-lag]) / n
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))
