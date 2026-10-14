"""A small, polite client for the free Open-Meteo APIs.

Endpoints used (checked September 2026, see open-meteo.com/en/docs):
- Forecast API        https://api.open-meteo.com/v1/forecast            live forecasts
- Previous Runs API   https://previous-runs-api.open-meteo.com/v1/forecast  past forecasts, by lead day
- Satellite API       https://satellite-api.open-meteo.com/v1/archive   satellite-derived irradiance
- Historical Weather  https://archive-api.open-meteo.com/v1/archive     reanalysis (ERA5 etc.)

Two conventions matter for solar work:
1. Radiation variables are the *mean of the preceding hour*: the value stamped 12:00 UTC
   covers 11:00–12:00 UTC. physics.py computes sun positions at the middle of the hour.
2. `<var>_previous_day1` is what the model predicted about 24 h before the valid time,
   `_previous_day2` about 48 h before, and so on.

The free tier is for non-commercial use (a portfolio project qualifies), needs no API key,
and allows fewer than 600 calls/minute, 5,000/hour and 10,000/day. A request with more than
10 variables or more than 2 weeks of data counts as several calls. Data are CC BY 4.0, so
credit "Weather data by Open-Meteo.com" in your README and website.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

ENDPOINTS = {
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "satellite": "https://satellite-api.open-meteo.com/v1/archive",
    "archive": "https://archive-api.open-meteo.com/v1/archive",
}

# Open-Meteo variable name -> the short column name used everywhere in SahelCast.
NWP_VARIABLES = {
    "shortwave_radiation": "ghi",  # global horizontal irradiance, W/m²
    "direct_normal_irradiance": "dni",  # W/m²
    "diffuse_radiation": "dhi",  # diffuse horizontal irradiance, W/m²
    "temperature_2m": "temp_air",  # °C
    "cloud_cover": "cloud_cover",  # %
    "wind_speed_10m": "wind_speed",  # m/s (we ask for wind_speed_unit=ms)
}
SATELLITE_VARIABLES = {
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
}
REANALYSIS_VARIABLES = {"temperature_2m": "temp_air", "wind_speed_10m": "wind_speed"}


class OpenMeteoError(RuntimeError):
    """The API answered, but not with usable data."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


class OpenMeteoClient:
    """HTTP client with a rate limit, retries and an on-disk cache of raw responses.

    The cache makes ingestion resumable: if a run stops halfway, the next run re-reads the
    responses it already has instead of calling the API again. It also lets you rebuild the
    lake from `data/raw` without any network access.
    """

    def __init__(
        self,
        transport: httpx.BaseTransport | None = None,
        requests_per_minute: int = 30,
        raw_dir: str | Path | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._http = httpx.Client(
            transport=transport,
            timeout=timeout,
            headers={"User-Agent": "sahelcast/0.1 (non-commercial portfolio project)"},
        )
        self._min_interval = 60.0 / max(requests_per_minute, 1)
        self._last_call = 0.0
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.calls = 0

    def __enter__(self) -> OpenMeteoClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def get(self, endpoint: str, params: dict, cache: bool = True) -> dict:
        path = self._cache_path(endpoint, params) if (cache and self.raw_dir) else None
        if path is not None and path.exists():
            return json.loads(path.read_text())
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        payload = self._fetch(ENDPOINTS[endpoint], params)
        self._last_call = time.monotonic()
        self.calls += 1
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
        return payload

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _fetch(self, url: str, params: dict) -> dict:
        resp = self._http.get(url, params=params)
        if resp.status_code == 400:  # bad request: retrying will not help
            reason = resp.json().get("reason", resp.text) if resp.content else resp.text
            raise OpenMeteoError(f"{url} rejected the request: {reason}")
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise OpenMeteoError(str(payload.get("reason")))
        return payload

    def _cache_path(self, endpoint: str, params: dict) -> Path:
        blob = json.dumps([endpoint, params], sort_keys=True, default=str).encode()
        key = hashlib.sha256(blob).hexdigest()[:20]
        assert self.raw_dir is not None
        return self.raw_dir / endpoint / f"{key}.json"


def date_chunks(start: date, end: date, days: int) -> Iterator[tuple[date, date]]:
    """Split [start, end] into consecutive windows of at most `days` days."""
    current = start
    while current <= end:
        stop = min(current + timedelta(days=days - 1), end)
        yield current, stop
        current = stop + timedelta(days=1)


def _base_params(lat: float, lon: float) -> dict:
    return {
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }


def hourly_frame(payload: dict, columns: dict[str, str], model: str | None = None) -> pd.DataFrame:
    """Turn the `hourly` block of a response into a DataFrame with UTC timestamps.

    When a request names models, some Open-Meteo endpoints suffix variable names with the
    model (for example `shortwave_radiation_ecmwf_ifs025`). Both spellings are accepted.
    """
    hourly = payload.get("hourly") or {}
    if "time" not in hourly:
        raise OpenMeteoError("the response has no hourly data")
    frame = pd.DataFrame({"valid_time": pd.to_datetime(hourly["time"], utc=True)})
    for api_name, short in columns.items():
        values = hourly.get(api_name)
        if values is None and model:
            values = hourly.get(f"{api_name}_{model}")
        if values is None:
            values = [None] * len(frame)
        frame[short] = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").astype(
            "float64"
        )
    return frame


def fetch_previous_runs(
    client: OpenMeteoClient,
    lat: float,
    lon: float,
    start: date,
    end: date,
    model: str,
    lead_days: list[int],
    cache: bool = True,
) -> tuple[pd.DataFrame, float | None]:
    """Past forecasts: one row per (valid_time, lead_day) with the NWP variables."""
    names = [f"{var}_previous_day{n}" for n in lead_days for var in NWP_VARIABLES]
    params = {
        **_base_params(lat, lon),
        "hourly": ",".join(names),
        "models": model,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    payload = client.get("previous_runs", params, cache=cache)
    frames = []
    for n in lead_days:
        mapping = {f"{var}_previous_day{n}": short for var, short in NWP_VARIABLES.items()}
        frame = hourly_frame(payload, mapping, model)
        frame.insert(1, "lead_day", n)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), payload.get("elevation")


def fetch_satellite(
    client: OpenMeteoClient,
    lat: float,
    lon: float,
    start: date,
    end: date,
    model: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Satellite-derived irradiance, used as the 'observed' truth."""
    params = {
        **_base_params(lat, lon),
        "hourly": ",".join(SATELLITE_VARIABLES),
        "models": model,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    return hourly_frame(client.get("satellite", params, cache=cache), SATELLITE_VARIABLES, model)


def fetch_reanalysis(
    client: OpenMeteoClient,
    lat: float,
    lon: float,
    start: date,
    end: date,
    model: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Observed-like air temperature and wind speed for the PV cell-temperature model."""
    params = {
        **_base_params(lat, lon),
        "hourly": ",".join(REANALYSIS_VARIABLES),
        "models": model,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    return hourly_frame(client.get("archive", params, cache=cache), REANALYSIS_VARIABLES, model)


def fetch_live_forecast(
    client: OpenMeteoClient, lat: float, lon: float, model: str, forecast_days: int = 3
) -> pd.DataFrame:
    """The latest forecast for the next days (never cached: it changes every run)."""
    params = {
        **_base_params(lat, lon),
        "hourly": ",".join(NWP_VARIABLES),
        "models": model,
        "forecast_days": forecast_days,
    }
    return hourly_frame(client.get("forecast", params, cache=False), NWP_VARIABLES, model)
