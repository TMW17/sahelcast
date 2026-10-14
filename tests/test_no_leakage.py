"""The most important tests in the project: a forecast may only use data available at
issue time. Each test corrupts everything observed after the cutoff and checks that the
forecasts do not change at all."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from sahelcast.baselines import forecast_lightgbm, forecast_persistence, forecast_physical
from sahelcast.config import Settings, min_lag_days
from sahelcast.dataset import Panel, build_panel, make_issues
from tests.conftest import synthetic_pv_hourly


@pytest.fixture
def panels(settings):
    pv, plants = synthetic_pv_hourly(days=90)
    panel_df = build_panel(pv, plants, lead=2)
    issue = make_issues(date(2025, 3, 15), date(2025, 3, 15), settings.forecast)[0]
    spoiled = panel_df.copy()
    future = spoiled["valid_time"] > issue.cutoff
    spoiled.loc[future, ["target_cf", "pv_actual_mw"]] = 0.999
    return Panel(panel_df), Panel(spoiled), issue


def test_the_schedule_is_what_the_docs_say(settings):
    issue = make_issues(date(2025, 3, 15), date(2025, 3, 15), settings.forecast)[0]
    assert issue.time == pd.Timestamp("2025-03-15 12:00", tz="UTC")
    # SARAH-3 arrives about two days late: observations end at 00:00 on the day before.
    assert issue.cutoff == pd.Timestamp("2025-03-14 00:00", tz="UTC")
    assert issue.target_start == pd.Timestamp("2025-03-16 00:00", tz="UTC")
    assert issue.target_end == pd.Timestamp("2025-03-16 23:00", tz="UTC")
    # Calibration may only learn from forecasts whose target day ended before the cutoff:
    # the one issued 3 days earlier (2025-03-12) targeted 2025-03-13.
    assert min_lag_days(settings.forecast) == settings.calibration.lag_days == 3
    too_soon = settings.model_dump()
    too_soon["calibration"]["lag_days"] = 2
    with pytest.raises(ValueError, match="lag_days >= 3"):
        Settings(**too_soon)


@pytest.mark.parametrize("model", [forecast_persistence, forecast_physical, forecast_lightgbm])
def test_forecasts_ignore_everything_after_the_cutoff(panels, settings, model):
    clean, spoiled, issue = panels
    settings.backtest.retrain_every_days = 1
    a = model(clean, [issue], settings)
    b = model(spoiled, [issue], settings)
    assert len(a) == len(b) > 0
    assert np.allclose(a.filter(regex=r"^q\d+$").to_numpy(), b.filter(regex=r"^q\d+$").to_numpy())


def test_chronos2_ignores_everything_after_the_cutoff(panels, settings):
    chronos = pytest.importorskip("chronos")
    import torch
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model

    from sahelcast.chronos_model import Chronos2Forecaster

    torch.manual_seed(0)
    cfg = Chronos2CoreConfig(d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=2)
    cfg.chronos_config = dict(
        context_length=1024,
        output_patch_size=16,
        input_patch_size=16,
        input_patch_stride=16,
        quantiles=[0.01, 0.05, *[round(x, 2) for x in np.arange(0.1, 0.91, 0.05)], 0.95, 0.99],
        use_reg_token=True,
        use_arcsinh=True,
        max_output_patches=64,
    )
    tiny = Chronos2Forecaster(pipeline=chronos.Chronos2Pipeline(Chronos2Model(cfg).eval()))
    clean, spoiled, issue = panels
    a, b = tiny.forecast(clean, [issue], settings), tiny.forecast(spoiled, [issue], settings)
    assert len(a) == 24 * len(clean.nodes)
    assert np.allclose(a.filter(regex=r"^q\d+$").to_numpy(), b.filter(regex=r"^q\d+$").to_numpy())


def test_aggregates_add_up_capacity(settings):
    pv, plants = synthetic_pv_hourly(days=5)
    panel = build_panel(pv, plants, lead=2)
    cap = panel.groupby("node_id")["ac_mw"].first()
    assert cap["DZ"] == pytest.approx(30.0)
    assert cap["DZ/W1"] == pytest.approx(30.0)
    assert cap["DZ/W1/a"] == pytest.approx(10.0)
