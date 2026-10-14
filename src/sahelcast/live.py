"""Tomorrow's forecast, produced every day at issue time (the operational run).

It reuses the backtest machinery on purpose: the models are run for the last `history_days`
issue days plus today, the calibration learns from the recent days whose outcomes are
known, and only today's forecast is kept. Tomorrow's weather covariates come from the latest
Forecast API run (bronze.nwp_live) where the history has no lead-2 values yet.

One honest caveat to state in your write-up: the live NWP run is fresher (and usually more
accurate) than the lead-2 forecasts the models were trained on, so live intervals tend to be
a little conservative.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .backtest import attach_truth, run_models
from .config import Settings
from .conformal import calibrate
from .dataset import Panel, build_panel, make_issues
from .lake import Lake
from .physics import pv_ac_mw, solar_geometry

log = logging.getLogger(__name__)


def live_rows(lake: Lake, settings: Settings) -> tuple[pd.DataFrame, pd.Timestamp]:
    """gold.pv_hourly-shaped rows computed from the latest live NWP run."""
    live = lake.sql(
        "SELECT * FROM lake.bronze.nwp_live "
        "WHERE issue_time = (SELECT max(issue_time) FROM lake.bronze.nwp_live)"
    )
    if live.empty:
        raise RuntimeError("bronze.nwp_live is empty: run `sahelcast ingest-live` first")
    plants = lake.read("silver.stg_plants")
    lead = settings.forecast.lead_day
    frames = []
    for plant in plants.itertuples():
        h = live[live["plant_id"] == plant.plant_id].sort_values("valid_time")
        idx = pd.DatetimeIndex(h["valid_time"])
        geom = solar_geometry(idx, plant.latitude, plant.longitude, plant.elevation_m)
        args = dict(
            geom=geom,
            tilt=plant.tilt_deg,
            azimuth=plant.azimuth_deg,
            dc_mw=plant.dc_mw,
            ac_mw=plant.ac_mw,
            pv=settings.pv,
        )
        frames.append(
            pd.DataFrame(
                {
                    "plant_id": plant.plant_id,
                    "valid_time": idx,
                    "solar_zenith": geom["apparent_zenith"].to_numpy(),
                    f"nwp{lead}_cloud_cover": h["cloud_cover"].to_numpy(),
                    f"nwp{lead}_temp_air": h["temp_air"].to_numpy(),
                    "ac_mw": plant.ac_mw,
                    "pv_actual_mw": np.nan,
                    f"pv_nwp{lead}_mw": pv_ac_mw(
                        h["ghi"], h["dni"], h["dhi"], h["temp_air"], h["wind_speed"], **args
                    ),
                    "pv_clearsky_mw": pv_ac_mw(
                        geom["clear_ghi"],
                        geom["clear_dni"],
                        geom["clear_dhi"],
                        h["temp_air"],
                        h["wind_speed"],
                        **args,
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True), pd.Timestamp(live["issue_time"].max())


def forecast_tomorrow(
    lake: Lake,
    settings: Settings,
    models: list[str] | None = None,
    history_days: int = 70,
    chronos=None,
) -> pd.DataFrame:
    fs, cal = settings.forecast, settings.calibration
    lead = fs.lead_day
    history = lake.read("gold.pv_hourly")
    live, fetched = live_rows(lake, settings)
    # Use history where it has lead-2 covariates; fill the remaining hours from the live run.
    have = history.dropna(subset=[f"pv_nwp{lead}_mw"])[["plant_id", "valid_time"]]
    extra = live.merge(have, on=["plant_id", "valid_time"], how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"].drop(columns="_merge")
    combined = pd.concat([history, extra], ignore_index=True)
    panel_df = build_panel(combined, lake.read("silver.stg_plants"), lead)
    today = fetched.normalize().date()
    issues = make_issues(today - pd.Timedelta(days=history_days), today, fs)
    raw = attach_truth(
        run_models(Panel(panel_df), issues, settings, models or fs.models, chronos), panel_df, lead
    )
    calibrated = calibrate(
        raw, fs.quantiles, cal.window_days, cal.lag_days, cal.gamma, cal.mondrian
    )
    out = pd.concat([raw, calibrated], ignore_index=True)
    out = out[out["issue_time"] == issues[-1].time].copy()
    out["nwp_fetched_at"] = fetched
    lake.upsert("gold.forecasts_live", out, ["model", "node_id", "issue_time", "valid_time"])
    log.info("forecast for %s issued at %s", issues[-1].target_start.date(), issues[-1].time)
    return out
