"""Collect data from Open-Meteo into the bronze layer of the lake.

Bronze tables (one row per plant and hour):
- bronze.plants              the plant registry (from config/plants.csv)
- bronze.site_elevation      ground elevation returned by the API (used by pvlib)
- bronze.nwp_previous_runs   past forecasts, one row per lead day (1 = ~24 h before, 2 = ~48 h)
- bronze.satellite           satellite-derived irradiance: the "observations"
- bronze.reanalysis          air temperature and wind speed
- bronze.nwp_live            the latest forecast, stamped with the time it was fetched
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from .config import Settings
from .fake_openmeteo import fake_transport
from .lake import Lake
from .openmeteo import (
    OpenMeteoClient,
    date_chunks,
    fetch_live_forecast,
    fetch_previous_runs,
    fetch_reanalysis,
    fetch_satellite,
)

log = logging.getLogger(__name__)

SOURCES = ("nwp", "satellite", "reanalysis")


def make_client(settings: Settings) -> OpenMeteoClient:
    """The real API, or the synthetic one when SAHELCAST_OFFLINE=1."""
    if settings.offline:
        return OpenMeteoClient(
            transport=fake_transport(),
            requests_per_minute=100_000,
            raw_dir=settings.paths.raw / "offline",
        )
    return OpenMeteoClient(
        requests_per_minute=settings.openmeteo.requests_per_minute, raw_dir=settings.paths.raw
    )


def load_registry(lake: Lake, plants: pd.DataFrame) -> None:
    cols = [
        "plant_id",
        "name",
        "country",
        "wilaya",
        "latitude",
        "longitude",
        "capacity_mw",
        "capacity_basis",
        "ac_mw",
        "dc_mw",
        "tilt_deg",
        "azimuth_deg",
        "status",
        "coord_quality",
        "commissioned",
    ]
    lake.upsert("bronze.plants", plants[[c for c in cols if c in plants.columns]], ["plant_id"])


def ingest_history(
    settings: Settings,
    plants: pd.DataFrame,
    lake: Lake,
    client: OpenMeteoClient,
    start: date | None = None,
    end: date | None = None,
    sources: tuple[str, ...] = SOURCES,
    recent_days: int = 10,
) -> dict[str, int]:
    """Download history for every plant, month by month, and upsert it into bronze."""
    om = settings.openmeteo
    start = start or om.history_start
    end = end or om.end()
    load_registry(lake, plants)
    counts = dict.fromkeys(sources, 0)
    elevations = []
    for plant in plants.itertuples():
        batches: dict[str, list[pd.DataFrame]] = {s: [] for s in sources}
        for chunk_start, chunk_end in date_chunks(start, end, om.chunk_days):
            # Recent days are still being filled in by the providers: never cache them.
            cache = chunk_end < date.today() - timedelta(days=recent_days)
            if "nwp" in sources:
                nwp, elevation = fetch_previous_runs(
                    client,
                    plant.latitude,
                    plant.longitude,
                    chunk_start,
                    chunk_end,
                    om.nwp_model,
                    om.lead_days,
                    cache=cache,
                )
                batches["nwp"].append(nwp.assign(model=om.nwp_model))
                if elevation is not None:
                    elevations.append({"plant_id": plant.plant_id, "elevation_m": float(elevation)})
            if "satellite" in sources:
                sat = fetch_satellite(
                    client,
                    plant.latitude,
                    plant.longitude,
                    chunk_start,
                    chunk_end,
                    om.satellite_model,
                    cache=cache,
                )
                batches["satellite"].append(sat.assign(model=om.satellite_model))
            if "reanalysis" in sources:
                rea = fetch_reanalysis(
                    client,
                    plant.latitude,
                    plant.longitude,
                    chunk_start,
                    chunk_end,
                    om.reanalysis_model,
                    cache=cache,
                )
                batches["reanalysis"].append(rea.assign(model=om.reanalysis_model))
        now = pd.Timestamp.now(tz="UTC")
        keys = {
            "nwp": ("bronze.nwp_previous_runs", ["plant_id", "model", "valid_time", "lead_day"]),
            "satellite": ("bronze.satellite", ["plant_id", "model", "valid_time"]),
            "reanalysis": ("bronze.reanalysis", ["plant_id", "model", "valid_time"]),
        }
        for source, frames in batches.items():
            if not frames:
                continue
            frame = pd.concat(frames, ignore_index=True)
            frame.insert(0, "plant_id", plant.plant_id)
            frame["ingested_at"] = now
            table, table_keys = keys[source]
            counts[source] += lake.upsert(table, frame, table_keys)
        log.info("ingested %s (%s to %s)", plant.plant_id, start, end)
    if elevations:
        lake.upsert("bronze.site_elevation", pd.DataFrame(elevations), ["plant_id"])
    log.info("Open-Meteo requests made: %s", client.calls)
    return counts


def ingest_live(
    settings: Settings, plants: pd.DataFrame, lake: Lake, client: OpenMeteoClient
) -> int:
    """Fetch the newest forecast for the next days. Run it daily at issue time."""
    issue_time = pd.Timestamp.now(tz="UTC").floor("h")
    frames = []
    for plant in plants.itertuples():
        frame = fetch_live_forecast(
            client, plant.latitude, plant.longitude, settings.openmeteo.nwp_model
        )
        frame.insert(0, "plant_id", plant.plant_id)
        frame["model"] = settings.openmeteo.nwp_model
        frame["issue_time"] = issue_time
        frames.append(frame)
    live = pd.concat(frames, ignore_index=True)
    live["ingested_at"] = pd.Timestamp.now(tz="UTC")
    return lake.upsert("bronze.nwp_live", live, ["plant_id", "model", "issue_time", "valid_time"])
