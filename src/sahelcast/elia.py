"""External benchmark on real, metered solar data: Elia, the Belgian grid operator.

Algerian plant output is not public, so SahelCast's "observations" are simulated. To show the
forecasting and calibration method also works on metered data, run the same pipeline for
Belgium and compare it with Elia's own published forecasts (scripts/benchmark_elia.py).

Dataset ods032 (historical, 15-minute values, MW) holds measured PV output for Belgium and
its regions, plus Elia's forecasts with 10 % and 90 % confidence bounds. The field names
below come from Elia's Explore API; check them before a long run:
    https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032
Elia's point forecast is treated as its median. Elia forecasts use fresher weather data than
SahelCast's lead-2 covariates, so the comparison is tilted in Elia's favour: say so.
"""

from __future__ import annotations

import io
from datetime import date

import httpx
import numpy as np
import pandas as pd

from .metrics import pinball

EXPORT_URL = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{dataset}/exports/parquet"
FIELDS = {
    "measured": "measured_mw",
    "dayahead11hforecast": "elia_q50",
    "dayahead11hconfidence10": "elia_q10",
    "dayahead11hconfidence90": "elia_q90",
    "monitoredcapacity": "capacity_mw",
}


def download_elia(
    start: date, end: date, region: str = "Belgium", dataset: str = "ods032", transport=None
) -> pd.DataFrame:
    """All rows for one region between start (inclusive) and end (exclusive)."""
    params = {
        "where": f"datetime >= date'{start.isoformat()}' AND datetime < date'{end.isoformat()}'",
        "refine": f'region:"{region}"',
        "timezone": "UTC",
    }
    with httpx.Client(transport=transport, timeout=300) as client:
        resp = client.get(EXPORT_URL.format(dataset=dataset), params=params)
        resp.raise_for_status()
    return pd.read_parquet(io.BytesIO(resp.content))


def to_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Average the quarter-hours and label each hour by its end, like Open-Meteo does."""
    missing = set(FIELDS) - set(raw.columns)
    if missing:
        raise ValueError(
            f"Elia fields not found: {sorted(missing)}. Available: {sorted(raw.columns)}"
        )
    df = raw[["datetime", *FIELDS]].rename(columns=FIELDS)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    hourly = df.set_index("datetime").sort_index().resample("h").mean()  # [H, H+1)
    hourly.index = hourly.index + pd.Timedelta(hours=1)
    return hourly.rename_axis("valid_time").reset_index()


def _scores(y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> dict[str, float]:
    return {
        "n_hours": int(len(y)),
        "pinball_q10": float(pinball(y, q10, 0.1).mean()),
        "pinball_q50": float(pinball(y, q50, 0.5).mean()),
        "pinball_q90": float(pinball(y, q90, 0.9).mean()),
        "mean_pinball": float(
            np.mean([pinball(y, q, t).mean() for q, t in ((q10, 0.1), (q50, 0.5), (q90, 0.9))])
        ),
        "coverage_80": float(np.mean((y >= q10) & (y <= q90))),
    }


def elia_scores(hourly: pd.DataFrame) -> dict[str, float]:
    """Elia's own forecast quality, in capacity-factor units, on daylight hours."""
    d = hourly[
        (hourly["capacity_mw"] > 0) & hourly["measured_mw"].notna() & (hourly["elia_q90"] > 0)
    ]
    cap = d["capacity_mw"].to_numpy()
    return _scores(
        d["measured_mw"].to_numpy() / cap,
        d["elia_q10"].to_numpy() / cap,
        d["elia_q50"].to_numpy() / cap,
        d["elia_q90"].to_numpy() / cap,
    )


def our_scores(forecasts: pd.DataFrame) -> dict[str, float]:
    """The same scores for SahelCast forecasts (columns y, q10, q50, q90, cf_clearsky)."""
    d = forecasts[forecasts["y"].notna() & (forecasts["cf_clearsky"] > 0.01)]
    return _scores(d["y"].to_numpy(), d["q10"].to_numpy(), d["q50"].to_numpy(), d["q90"].to_numpy())


def replace_target(panel_df: pd.DataFrame, hourly: pd.DataFrame, node: str = "BE") -> pd.DataFrame:
    """Use Elia's metered capacity factor as the observation for the national node."""
    cf = (
        hourly.set_index("valid_time")["measured_mw"]
        / hourly.set_index("valid_time")["capacity_mw"]
    ).clip(0, 1)
    panel_df = panel_df.copy()
    rows = panel_df["node_id"] == node
    values = panel_df.loc[rows, "valid_time"].map(cf).to_numpy(dtype=float)
    panel_df.loc[rows, "target_cf"] = values
    panel_df.loc[rows, "pv_actual_mw"] = values * panel_df.loc[rows, "ac_mw"].to_numpy()
    return panel_df
