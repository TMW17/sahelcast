"""Physics with pvlib: sun position, clear-sky irradiance and PV power.

The same PV model turns three kinds of weather into power:
- satellite irradiance + reanalysis temperature -> pv_actual_mw  (the target, "virtual telemetry")
- NWP irradiance + NWP temperature               -> pv_nwp1_mw, pv_nwp2_mw (physical forecasts)
- clear-sky irradiance + NWP temperature         -> pv_clearsky_mw (what a cloudless sky gives)

Algerian plant output is not public, so pv_actual_mw is a simulation. The guide shows how to
check the simulation method on real plants (NREL PVDAQ) and how to benchmark the forecasting
method on real grid data (Elia, Belgium).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from .config import PVSettings, Settings
from .lake import Lake

HALF_HOUR = pd.Timedelta(minutes=30)


def solar_geometry(
    valid_time: pd.DatetimeIndex, lat: float, lon: float, altitude: float
) -> pd.DataFrame:
    """Sun position and clear-sky irradiance for hour-ending timestamps.

    Open-Meteo radiation is the mean of the preceding hour, so the sun is placed at the
    middle of that hour (valid_time - 30 min). Using the end of the hour instead shifts the
    whole day by 30 minutes and adds large errors at sunrise and sunset.
    """
    mid = valid_time - HALF_HOUR
    loc = pvlib.location.Location(lat, lon, tz="UTC", altitude=altitude)
    solpos = loc.get_solarposition(mid)
    clear = loc.get_clearsky(mid, model="ineichen", solar_position=solpos)
    return pd.DataFrame(
        {
            "apparent_zenith": solpos["apparent_zenith"].to_numpy(),
            "azimuth": solpos["azimuth"].to_numpy(),
            "clear_ghi": clear["ghi"].to_numpy(),
            "clear_dni": clear["dni"].to_numpy(),
            "clear_dhi": clear["dhi"].to_numpy(),
            "dni_extra": pvlib.irradiance.get_extra_radiation(mid).to_numpy(),
        },
        index=valid_time,
    )


def pv_ac_mw(
    ghi: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    temp_air: np.ndarray,
    wind_speed: np.ndarray,
    geom: pd.DataFrame,
    tilt: float,
    azimuth: float,
    dc_mw: float,
    ac_mw: float,
    pv: PVSettings,
) -> np.ndarray:
    """AC power in MW for a fixed-tilt plant (PVWatts model). NaN where irradiance is missing."""
    ghi, dni, dhi = (np.asarray(x, dtype=float) for x in (ghi, dni, dhi))
    missing = np.isnan(ghi) | np.isnan(dni) | np.isnan(dhi)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,  # pvlib: 180 = south (Open-Meteo would say 0 = south)
        solar_zenith=geom["apparent_zenith"].to_numpy(),
        solar_azimuth=geom["azimuth"].to_numpy(),
        dni=np.nan_to_num(dni),
        ghi=np.nan_to_num(ghi),
        dhi=np.nan_to_num(dhi),
        dni_extra=geom["dni_extra"].to_numpy(),
        albedo=pv.albedo,
        model=pv.transposition_model,
    )
    poa_global = np.clip(np.nan_to_num(np.asarray(poa["poa_global"], dtype=float)), 0, None)
    temp = np.nan_to_num(np.asarray(temp_air, dtype=float), nan=25.0)
    wind = np.nan_to_num(np.asarray(wind_speed, dtype=float), nan=2.0)
    temp_cell = pvlib.temperature.faiman(poa_global, temp, wind)
    pdc = pvlib.pvsystem.pvwatts_dc(poa_global, temp_cell, pdc0=dc_mw, gamma_pdc=pv.gamma_pdc)
    pdc = np.asarray(pdc, dtype=float) * (1 - pv.system_losses)
    pac = np.asarray(pvlib.inverter.pvwatts(pdc, pdc0=ac_mw / 0.96, eta_inv_nom=0.96), dtype=float)
    pac = np.clip(np.nan_to_num(pac), 0, ac_mw)
    pac[geom["apparent_zenith"].to_numpy() >= 90] = 0.0
    pac[missing] = np.nan
    return pac


def plant_pv_table(hourly: pd.DataFrame, plant: pd.Series, settings: Settings) -> pd.DataFrame:
    """All PV columns for one plant. `hourly` is that plant's rows of gold.fct_hourly."""
    pv, lead = settings.pv, settings.forecast.lead_day
    h = hourly.sort_values("valid_time").reset_index(drop=True)
    idx = pd.DatetimeIndex(h["valid_time"])
    geom = solar_geometry(idx, plant.latitude, plant.longitude, plant.elevation_m)
    args = dict(
        geom=geom,
        tilt=plant.tilt_deg,
        azimuth=plant.azimuth_deg,
        dc_mw=plant.dc_mw,
        ac_mw=plant.ac_mw,
        pv=pv,
    )
    out = pd.DataFrame(
        {
            "plant_id": plant.plant_id,
            "valid_time": idx,
            "solar_zenith": geom["apparent_zenith"].to_numpy(),
            "clear_ghi": geom["clear_ghi"].to_numpy(),
            "obs_ghi": h["obs_ghi"].to_numpy(),
            f"nwp{lead}_ghi": h[f"nwp{lead}_ghi"].to_numpy(),
            f"nwp{lead}_cloud_cover": h[f"nwp{lead}_cloud_cover"].to_numpy(),
            f"nwp{lead}_temp_air": h[f"nwp{lead}_temp_air"].to_numpy(),
            "ac_mw": plant.ac_mw,
        }
    )
    # Reanalysis temperature and wind arrive a few days late. Until then, use the day-ahead
    # forecast's values: through the cell temperature they change the output by a few percent.
    first = min(settings.openmeteo.lead_days)
    obs_temp = h["obs_temp_air"].fillna(h[f"nwp{first}_temp_air"])
    obs_wind = h["obs_wind_speed"].fillna(h[f"nwp{first}_wind_speed"])
    actual = pv_ac_mw(h["obs_ghi"], h["obs_dni"], h["obs_dhi"], obs_temp, obs_wind, **args)
    # A satellite value far above the clear-sky value is a retrieval error, not sunshine.
    implausible = h["obs_ghi"].to_numpy() > 1.2 * geom["clear_ghi"].to_numpy() + 50
    actual[implausible] = np.nan
    out["pv_actual_mw"] = actual
    for n in settings.openmeteo.lead_days:
        out[f"pv_nwp{n}_mw"] = pv_ac_mw(
            h[f"nwp{n}_ghi"],
            h[f"nwp{n}_dni"],
            h[f"nwp{n}_dhi"],
            h[f"nwp{n}_temp_air"],
            h[f"nwp{n}_wind_speed"],
            **args,
        )
    out["pv_clearsky_mw"] = pv_ac_mw(
        geom["clear_ghi"],
        geom["clear_dni"],
        geom["clear_dhi"],
        h[f"nwp{lead}_temp_air"],
        h[f"nwp{lead}_wind_speed"],
        **args,
    )
    return out


def build_pv_hourly(lake: Lake, settings: Settings) -> pd.DataFrame:
    """Compute gold.pv_hourly for every plant and write it to the lake."""
    hourly = lake.read("gold.fct_hourly")
    plants = lake.read("silver.stg_plants")
    frames = [
        plant_pv_table(hourly[hourly["plant_id"] == plant.plant_id], plant, settings)
        for plant in plants.itertuples()
    ]
    table = pd.concat(frames, ignore_index=True)
    lake.replace("gold.pv_hourly", table)
    return table


def daily_regimes(frame: pd.DataFrame, key: str = "node_id") -> pd.DataFrame:
    """Label each day by its observed clear-sky index: clear, mixed or overcast/dusty.

    Only for evaluation (it uses observations of the target day). In the Sahara a low index
    on a cloudless day usually means dust.
    """
    day = (frame["valid_time"] - HALF_HOUR).dt.floor("D")
    g = frame.assign(day=day).groupby([key, "day"])
    csi = g["pv_actual_mw"].sum(min_count=1) / g["pv_clearsky_mw"].sum()
    out = csi.rename("csi_day").reset_index()
    out["regime"] = pd.cut(
        out["csi_day"], [-np.inf, 0.6, 0.85, np.inf], labels=["overcast_or_dust", "mixed", "clear"]
    ).astype("object")
    return out
