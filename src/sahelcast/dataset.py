"""Model-ready panel of series and the leak-free forecast schedule.

Series ("nodes") form a hierarchy, named like hierarchicalforecast does:
    DZ                         the national total
    DZ/Djelfa                  a wilaya (sum of its plants)
    DZ/Djelfa/ain_el_ibel_1    a plant
Every node has the same columns, expressed as capacity factors (MW divided by AC capacity),
so one model can forecast plants, wilayas and the country.

The forecast schedule is where leakage is prevented:
- a forecast is issued at 12:00 UTC on day D for the 24 hours of day D+1
- observations are only used up to `cutoff` (36 h before issue by default, because the
  SARAH-3 satellite data are published about two days late)
- weather covariates come from `*_previous_day2` forecasts, which already existed at issue time
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .config import ForecastSettings

MW_COLUMNS = ["pv_actual_mw", "pv_clearsky_mw", "pv_nwp1_mw", "pv_nwp2_mw", "ac_mw"]


def qcol(q: float) -> str:
    """Column name for a quantile: 0.05 -> 'q05', 0.5 -> 'q50'."""
    return f"q{round(q * 100):02d}"


def covariate_columns(lead: int) -> list[str]:
    return [
        f"cf_nwp{lead}",
        "cf_clearsky",
        f"nwp{lead}_cloud_cover",
        f"nwp{lead}_temp_air",
        "solar_zenith",
    ]


def build_panel(pv_hourly: pd.DataFrame, plants: pd.DataFrame, lead: int) -> pd.DataFrame:
    """Plant rows plus capacity-weighted aggregates for every wilaya and the whole country."""
    plants = plants.assign(country=plants.get("country", "DZ"))
    p = pv_hourly.merge(
        plants[["plant_id", "wilaya", "latitude", "country"]], on="plant_id", how="inner"
    )
    p["plant_node"] = p["country"] + "/" + p["wilaya"] + "/" + p["plant_id"]
    p["wilaya_node"] = p["country"] + "/" + p["wilaya"]
    p["country_node"] = p["country"]
    mw_cols = [c for c in MW_COLUMNS if c in p.columns]
    weighted = [f"nwp{lead}_cloud_cover", f"nwp{lead}_temp_air", "solar_zenith", "latitude"]

    frames = []
    for level, key in (
        ("plant", "plant_node"),
        ("wilaya", "wilaya_node"),
        ("country", "country_node"),
    ):
        g = p.groupby([key, "valid_time"])
        out = g[mw_cols].sum(min_count=1)
        # The aggregate is only "observed" when every plant in it is observed.
        complete = g["pv_actual_mw"].count() == g["pv_actual_mw"].size()
        out.loc[~complete, "pv_actual_mw"] = float("nan")
        for col in weighted:
            out[col] = (p[col] * p["ac_mw"]).groupby([p[key], p["valid_time"]]).sum() / out["ac_mw"]
        out = out.reset_index().rename(columns={key: "node_id"})
        out["level"] = level
        frames.append(out)
    panel = pd.concat(frames, ignore_index=True)
    panel["target_cf"] = panel["pv_actual_mw"] / panel["ac_mw"]
    panel["cf_clearsky"] = panel["pv_clearsky_mw"] / panel["ac_mw"]
    for n in (1, 2):
        if f"pv_nwp{n}_mw" in panel:
            panel[f"cf_nwp{n}"] = panel[f"pv_nwp{n}_mw"] / panel["ac_mw"]
    return panel.sort_values(["node_id", "valid_time"]).reset_index(drop=True)


@dataclass(frozen=True)
class Issue:
    time: pd.Timestamp  # when the forecast is made (UTC)
    cutoff: pd.Timestamp  # last hour whose observation the forecast may use
    target_start: pd.Timestamp  # first hour forecast (hour-ending label, UTC)
    target_end: pd.Timestamp  # last hour forecast

    @property
    def day(self) -> date:
        return self.time.date()


def make_issues(start: date, end: date, fs: ForecastSettings) -> list[Issue]:
    issues = []
    for d in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D", tz="UTC"):
        t = d + pd.Timedelta(hours=fs.issue_hour_utc)
        target_start = d + pd.Timedelta(days=1)
        issues.append(
            Issue(
                time=t,
                cutoff=t - pd.Timedelta(hours=fs.obs_cutoff_hours),
                target_start=target_start,
                target_end=target_start + pd.Timedelta(hours=23),
            )
        )
    return issues


class Panel:
    """Fast time slicing of the panel, one time-indexed frame per node."""

    def __init__(self, panel: pd.DataFrame) -> None:
        self.df = panel
        self.frames = {
            node: g.set_index("valid_time").sort_index() for node, g in panel.groupby("node_id")
        }
        self.nodes = sorted(self.frames)
        self.level = panel.groupby("node_id")["level"].first().to_dict()

    def history(self, node: str, cutoff: pd.Timestamp, days: int) -> pd.DataFrame:
        """Rows in (cutoff - days, cutoff]: everything a forecast may learn from."""
        frame = self.frames[node]
        return frame.loc[cutoff - pd.Timedelta(days=days) + pd.Timedelta(hours=1) : cutoff]

    def window(self, node: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return self.frames[node].loc[start:end]
