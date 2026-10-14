import numpy as np
import pandas as pd
import pytest

from sahelcast.conformal import calibrate
from sahelcast.dataset import qcol
from sahelcast.metrics import add_losses, diebold_mariano, pinball, summarize

QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def biased_overconfident_forecasts(days: int = 150, seed: int = 1) -> pd.DataFrame:
    """Forecasts whose median is 0.05 too low and whose ranges are half as wide as they should be."""
    rng = np.random.default_rng(seed)
    issue = np.repeat(pd.date_range("2025-01-01 12:00", periods=days, freq="D", tz="UTC"), 24)
    valid = issue + pd.to_timedelta(np.tile(np.arange(12, 36), days), unit="h")
    clear = np.tile(np.clip(np.sin((np.arange(24) - 6) / 12 * np.pi), 0, None), days)
    y = np.clip(0.6 * clear + rng.normal(0, 0.08, len(clear)) * clear, 0, 1)
    median = np.clip(0.6 * clear - 0.05 * clear, 0, 1)
    z = {
        0.05: -1.645,
        0.1: -1.2816,
        0.25: -0.6745,
        0.5: 0.0,
        0.75: 0.6745,
        0.9: 1.2816,
        0.95: 1.645,
    }
    frame = pd.DataFrame(
        {
            "model": "m",
            "node_id": "DZ",
            "issue_time": issue,
            "valid_time": valid,
            "y": y,
            "cf_clearsky": clear,
            "cf_nwp": median,
        }
    )
    for q in QUANTILES:
        frame[qcol(q)] = np.clip(median + z[q] * 0.04 * clear, 0, 1)
    return frame


def test_calibration_fixes_bias_and_coverage():
    fc = biased_overconfident_forecasts()
    cal = calibrate(fc, QUANTILES, window_days=60, lag_days=2, gamma=0.005)
    late = cal["issue_time"] >= cal["issue_time"].min() + pd.Timedelta(days=70)
    raw_scores = summarize(add_losses(fc[late.to_numpy()], QUANTILES), ["model"]).iloc[0]
    cal_scores = summarize(add_losses(cal[late], QUANTILES), ["model"]).iloc[0]
    assert raw_scores["coverage_80"] < 0.6
    assert abs(cal_scores["coverage_80"] - 0.8) < 0.05
    assert cal_scores["quantile_score"] < raw_scores["quantile_score"]
    day = cal[late & (cal["cf_clearsky"] > 0.01)]
    assert abs((day["y"] < day["q50"]).mean() - 0.5) < 0.06


def test_calibration_never_uses_outcomes_that_are_not_known_yet():
    fc = biased_overconfident_forecasts(days=90)
    last_two = fc["issue_time"] >= fc["issue_time"].max() - pd.Timedelta(days=1)
    changed = fc.copy()
    changed.loc[last_two, "y"] = 0.0  # outcomes of the last two issue days
    a = calibrate(fc, QUANTILES, 60, lag_days=2, gamma=0.005)
    b = calibrate(changed, QUANTILES, 60, lag_days=2, gamma=0.005)
    cols = [qcol(q) for q in QUANTILES]
    assert np.allclose(a[cols].to_numpy(), b[cols].to_numpy())


def test_pinball_loss_values():
    assert pinball(np.array([1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.9)
    assert pinball(np.array([0.0]), np.array([1.0]), 0.9)[0] == pytest.approx(0.1)


def test_diebold_mariano_detects_the_better_model():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.random(200) + 1.0)
    stat, p = diebold_mariano(base - 0.3, base + rng.normal(0, 0.05, 200))
    assert stat < 0 and p < 0.01
