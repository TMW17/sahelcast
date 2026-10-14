"""A synthetic stand-in for the Open-Meteo APIs, for tests, CI and `sahelcast demo`.

It answers the same URLs with JSON of exactly the same shape as the real services, so the
whole pipeline (parsing, lakehouse, dbt, models) runs without internet access. The weather
is invented: pvlib clear-sky irradiance multiplied by a cloud-and-dust attenuation that has
a regional part (shared by all plants) and a local part. "Forecasts" are the truth plus
errors that grow with lead time and that underestimate dust, like real NWP models often do.

None of these numbers are real. Never report results computed on them.
"""

from __future__ import annotations

import zlib
from datetime import date, timedelta
from functools import lru_cache

import httpx
import numpy as np
import pandas as pd
import pvlib

_EPOCH = date(2020, 1, 1)
_N_DAYS = 365 * 12


@lru_cache(maxsize=256)
def _ar1(seed: int, rho: float, sigma: float) -> np.ndarray:
    """A daily AR(1) series from 2020 to 2031, reproducible from its seed."""
    rng = np.random.default_rng(seed)
    x = np.empty(_N_DAYS)
    x[0] = 0.0
    shocks = rng.standard_normal(_N_DAYS) * sigma
    for i in range(1, _N_DAYS):
        x[i] = rho * x[i - 1] + shocks[i]
    return x


def _site_seed(lat: float, lon: float) -> int:
    return zlib.crc32(f"{lat:.2f},{lon:.2f}".encode()) % 100_000


def _truth(lat: float, lon: float, times: pd.DatetimeIndex) -> pd.DataFrame:
    """The invented 'true' weather at one site, hourly (timestamps = end of hour)."""
    mid = times - pd.Timedelta(minutes=30)
    loc = pvlib.location.Location(lat, lon, tz="UTC", altitude=_elevation(lat, lon))
    solpos = loc.get_solarposition(mid)
    clear = loc.get_clearsky(mid, model="ineichen", solar_position=solpos)
    day = np.clip((mid.normalize().tz_localize(None) - pd.Timestamp(_EPOCH)).days, 0, _N_DAYS - 1)
    day = np.asarray(day)
    seed = _site_seed(lat, lon)

    # Cloudiness: wetter north in winter, a mostly clear Sahara.
    doy = mid.dayofyear.to_numpy()
    winter = np.cos(2 * np.pi * (doy - 15) / 365.25)  # +1 in January, -1 in July
    northness = np.clip((lat - 27.0) / 9.0, 0, 1)
    base = -2.2 + 1.6 * northness * (0.5 + 0.5 * winter)
    regional = _ar1(1, 0.7, 0.8)[day]
    local = _ar1(seed, 0.5, 0.6)[day]
    hourly = np.random.default_rng(seed + 7).standard_normal(len(times)) * 0.35
    cloud_frac = 1.0 / (1.0 + np.exp(-(base + regional + local + hourly)))

    # Dust events: multi-day episodes, stronger in spring and in the south.
    dust_driver = _ar1(3, 0.85, 0.5)[day] + 0.4 * np.sin(2 * np.pi * (doy - 60) / 365.25)
    dust = np.clip(dust_driver - 0.9, 0, None) * (1.2 - 0.5 * northness)
    dust_factor = np.clip(1.0 - dust, 0.55, 1.0)

    k = np.clip(1.0 - 0.85 * cloud_frac, 0.08, 1.0) * dust_factor
    ghi = clear["ghi"].to_numpy() * k
    zenith = solpos["zenith"].to_numpy()
    parts = pvlib.irradiance.erbs(ghi, zenith, mid)
    temp = (
        24
        - 10 * northness * (0.5 + 0.5 * winter)
        - 6 * winter
        + 7 * np.sin(2 * np.pi * (mid.hour.to_numpy() - 9) / 24)
    )
    wind = np.clip(3.5 + 1.5 * _ar1(seed + 11, 0.6, 1.0)[day], 0.3, None)
    return pd.DataFrame(
        {
            "k": k,
            "cloud_frac": cloud_frac,
            "dust_factor": dust_factor,
            "ghi": ghi,
            "dni": np.nan_to_num(parts["dni"]),
            "dhi": np.nan_to_num(parts["dhi"]),
            "temp_air": temp,
            "wind_speed": wind,
            "clear_ghi": clear["ghi"].to_numpy(),
            "zenith": zenith,
        },
        index=times,
    )


def _forecast(lat: float, lon: float, times: pd.DatetimeIndex, lead: int) -> pd.DataFrame:
    """Truth plus a forecast error that grows with lead time and misses most dust."""
    truth = _truth(lat, lon, times)
    mid = times - pd.Timedelta(minutes=30)
    day = np.clip((mid.normalize().tz_localize(None) - pd.Timestamp(_EPOCH)).days, 0, _N_DAYS - 1)
    day = np.asarray(day)
    seed = _site_seed(lat, lon)
    err = _ar1(seed + 100 + lead, 0.4, 0.05 + 0.04 * lead)[day]
    err = err + np.random.default_rng(seed + 200 + lead).standard_normal(len(times)) * 0.04
    seen_dust = truth["dust_factor"].to_numpy() ** 0.3  # the model sees only part of the dust
    k = np.clip(
        truth["k"].to_numpy() / truth["dust_factor"].to_numpy() * seen_dust + err, 0.05, 1.05
    )
    ghi = truth["clear_ghi"].to_numpy() * k
    parts = pvlib.irradiance.erbs(ghi, truth["zenith"].to_numpy(), mid)
    cloud = np.clip(truth["cloud_frac"].to_numpy() - err, 0, 1) * 100
    return pd.DataFrame(
        {
            "ghi": ghi,
            "dni": np.nan_to_num(parts["dni"]),
            "dhi": np.nan_to_num(parts["dhi"]),
            "temp_air": truth["temp_air"].to_numpy() + 0.5 * lead * err * 10,
            "cloud_cover": cloud,
            "wind_speed": truth["wind_speed"].to_numpy() * (1 + err),
        },
        index=times,
    )


def _elevation(lat: float, lon: float) -> float:
    return float(round(400 + 600 * abs(np.sin(np.radians(lat * 7 + lon * 3))), 0))


_SHORT = {
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
    "temperature_2m": "temp_air",
    "cloud_cover": "cloud_cover",
    "wind_speed_10m": "wind_speed",
}
_UNITS = {
    "shortwave_radiation": "W/m²",
    "direct_normal_irradiance": "W/m²",
    "diffuse_radiation": "W/m²",
    "temperature_2m": "°C",
    "cloud_cover": "%",
    "wind_speed_10m": "m/s",
}


def _times(params: httpx.QueryParams) -> pd.DatetimeIndex:
    if "start_date" in params:
        start = pd.Timestamp(params["start_date"], tz="UTC")
        end = pd.Timestamp(params["end_date"], tz="UTC") + pd.Timedelta(hours=23)
    else:
        start = pd.Timestamp.now(tz="UTC").normalize()
        end = start + pd.Timedelta(days=int(params.get("forecast_days", 7))) - pd.Timedelta(hours=1)
    return pd.date_range(start, end, freq="h")


def _payload(host: str, params: httpx.QueryParams) -> dict:
    lat, lon = float(params["latitude"]), float(params["longitude"])
    times = _times(params)
    hourly: dict[str, list] = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    units: dict[str, str] = {"time": "iso8601"}
    cache: dict[int, pd.DataFrame] = {}
    for name in params["hourly"].split(","):
        base, lead = name, 0
        if "_previous_day" in name:
            base, n = name.split("_previous_day")
            lead = int(n)
        if host.startswith(("satellite-api", "archive-api")):
            frame = cache.setdefault(-1, _truth(lat, lon, times))
            if "cloud_cover" in base:
                frame = frame.assign(cloud_cover=frame["cloud_frac"] * 100)
        else:
            frame = cache.setdefault(lead, _forecast(lat, lon, times, max(lead, 0)))
        values = frame[_SHORT[base]].to_numpy().astype(float)
        if host.startswith("satellite-api"):  # satellites miss the odd hour
            gaps = np.random.default_rng(_site_seed(lat, lon) + 5).random(len(values)) < 0.005
            values = np.where(gaps, np.nan, values)
        hourly[name] = [None if np.isnan(v) else round(float(v), 1) for v in values]
        units[name] = _UNITS[base]
    return {
        "latitude": lat,
        "longitude": lon,
        "generationtime_ms": 0.1,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "timezone_abbreviation": "GMT",
        "elevation": _elevation(lat, lon),
        "hourly_units": units,
        "hourly": hourly,
    }


def fake_transport() -> httpx.MockTransport:
    """An httpx transport that answers Open-Meteo URLs with synthetic data."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_payload(request.url.host, request.url.params))

    return httpx.MockTransport(handler)


def demo_window(days: int = 150) -> tuple[date, date]:
    """A recent date window for the offline demo."""
    end = date.today() - timedelta(days=7)
    return end - timedelta(days=days - 1), end
