import io

import httpx
import numpy as np
import pandas as pd

from sahelcast.elia import download_elia, elia_scores, replace_target, to_hourly


def fake_elia(hours: int = 48) -> pd.DataFrame:
    t = pd.date_range("2025-06-01", periods=hours * 4, freq="15min", tz="UTC")
    sun = np.clip(np.sin((t.hour + t.minute / 60 - 6) / 12 * np.pi), 0, None)
    return pd.DataFrame(
        {
            "datetime": t,
            "region": "Belgium",
            "measured": 5000 * sun,
            "dayahead11hforecast": 4800 * sun,
            "dayahead11hconfidence10": 4000 * sun,
            "dayahead11hconfidence90": 5600 * sun,
            "monitoredcapacity": 10000.0,
        }
    )


def test_download_and_hourly_labels_match_open_meteo():
    buffer = io.BytesIO()
    fake_elia().to_parquet(buffer)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=buffer.getvalue()))
    hourly = to_hourly(
        download_elia(
            pd.Timestamp("2025-06-01").date(),
            pd.Timestamp("2025-06-03").date(),
            transport=transport,
        )
    )
    assert len(hourly) == 48
    # The hour labelled 13:00 is the mean of the quarter-hours 12:00, 12:15, 12:30, 12:45.
    raw = fake_elia().set_index("datetime")["measured"]
    expected = raw["2025-06-01 12:00":"2025-06-01 12:45"].mean()
    assert np.isclose(
        hourly.set_index("valid_time").loc["2025-06-01 13:00", "measured_mw"], expected
    )
    scores = elia_scores(hourly)
    assert scores["coverage_80"] == 1.0 and scores["n_hours"] > 0


def test_replace_target_uses_metered_capacity_factor():
    hourly = to_hourly(fake_elia())
    panel = pd.DataFrame(
        {
            "node_id": "BE",
            "valid_time": hourly["valid_time"],
            "ac_mw": 600.0,
            "target_cf": 0.0,
            "pv_actual_mw": 0.0,
        }
    )
    out = replace_target(panel, hourly)
    assert np.allclose(out["target_cf"], (hourly["measured_mw"] / 10000.0).to_numpy())
    assert np.allclose(out["pv_actual_mw"], out["target_cf"] * 600.0)
