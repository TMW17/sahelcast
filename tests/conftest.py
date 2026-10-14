from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sahelcast.config import load_plants, load_settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Settings pointing at a temporary folder, with the synthetic Open-Meteo API."""
    monkeypatch.setenv("SAHELCAST_OFFLINE", "1")
    s = load_settings()
    s.paths.lake_metadata = tmp_path / "lake" / "metadata.ducklake"
    s.paths.lake_data = tmp_path / "lake" / "files"
    s.paths.raw = tmp_path / "raw"
    s.paths.artifacts = tmp_path / "artifacts"
    s.paths.site = tmp_path / "site"
    return s


@pytest.fixture
def plants(settings):
    return load_plants(pv=settings.pv)


def synthetic_pv_hourly(days: int = 90, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small gold.pv_hourly-like table for two plants, built without the lake."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01 01:00", periods=days * 24, freq="h", tz="UTC")
    hour = (times - pd.Timedelta(minutes=30)).hour.to_numpy() + 0.5
    clear = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    plants = pd.DataFrame(
        {
            "plant_id": ["a", "b"],
            "wilaya": ["W1", "W1"],
            "latitude": [30.0, 31.0],
            "ac_mw": [10.0, 20.0],
        }
    )
    rows = []
    for p in plants.itertuples():
        daily = np.repeat(np.clip(1 - 0.4 * rng.random(days), 0.2, 1.0), 24)
        actual = clear * daily * p.ac_mw * 0.9
        nwp = actual * np.clip(1 + 0.15 * np.repeat(rng.standard_normal(days), 24), 0.3, 1.6)
        rows.append(
            pd.DataFrame(
                {
                    "plant_id": p.plant_id,
                    "valid_time": times,
                    "solar_zenith": np.where(clear > 0, 90 - 60 * clear, 100),
                    "nwp2_cloud_cover": (1 - daily) * 100,
                    "nwp2_temp_air": 20.0,
                    "ac_mw": p.ac_mw,
                    "pv_actual_mw": actual,
                    "pv_clearsky_mw": clear * p.ac_mw * 0.95,
                    "pv_nwp1_mw": nwp,
                    "pv_nwp2_mw": nwp,
                }
            )
        )
    return pd.concat(rows, ignore_index=True), plants
