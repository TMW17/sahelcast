import numpy as np
import pandas as pd
import pvlib

from sahelcast.config import PVSettings
from sahelcast.physics import pv_ac_mw, solar_geometry


def _day(lat=33.7, lon=2.8):
    times = pd.date_range("2025-06-21 01:00", periods=24, freq="h", tz="UTC")
    return times, solar_geometry(times, lat, lon, 750)


def test_sun_is_placed_in_the_middle_of_each_hour():
    times, geom = _day()
    loc = pvlib.location.Location(33.7, 2.8, altitude=750)
    expected = loc.get_solarposition(times - pd.Timedelta(minutes=30))["apparent_zenith"].to_numpy()
    assert np.allclose(geom["apparent_zenith"].to_numpy(), expected)


def test_clear_sky_power_is_zero_at_night_and_below_capacity():
    _, geom = _day()
    power = pv_ac_mw(
        geom["clear_ghi"],
        geom["clear_dni"],
        geom["clear_dhi"],
        np.full(24, 30.0),
        np.full(24, 3.0),
        geom,
        tilt=30,
        azimuth=180,
        dc_mw=24,
        ac_mw=20,
        pv=PVSettings(),
    )
    night = geom["apparent_zenith"].to_numpy() >= 90
    assert (power[night] == 0).all()
    assert power.max() <= 20.0
    assert power.max() > 14.0  # a clear June day in Laghouat gets close to full output


def test_missing_irradiance_gives_missing_power():
    _, geom = _day()
    ghi = geom["clear_ghi"].to_numpy().copy()
    ghi[12] = np.nan
    power = pv_ac_mw(
        ghi,
        geom["clear_dni"],
        geom["clear_dhi"],
        np.full(24, 25.0),
        np.full(24, 2.0),
        geom,
        30,
        180,
        12,
        10,
        PVSettings(),
    )
    assert np.isnan(power[12])
    assert np.isfinite(np.delete(power, 12)).all()
