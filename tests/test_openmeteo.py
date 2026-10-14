from datetime import date

import httpx
import pytest

from sahelcast.fake_openmeteo import fake_transport
from sahelcast.openmeteo import (
    OpenMeteoClient,
    OpenMeteoError,
    date_chunks,
    fetch_previous_runs,
    fetch_satellite,
    hourly_frame,
)


def test_date_chunks_cover_the_range_without_gaps():
    chunks = list(date_chunks(date(2025, 1, 1), date(2025, 3, 10), 31))
    assert chunks[0][0] == date(2025, 1, 1)
    assert chunks[-1][1] == date(2025, 3, 10)
    for (_, end), (start, _) in zip(chunks, chunks[1:], strict=False):
        assert (start - end).days == 1


def test_previous_runs_are_parsed_one_row_per_hour_and_lead_day():
    with OpenMeteoClient(transport=fake_transport(), requests_per_minute=10_000) as client:
        frame, elevation = fetch_previous_runs(
            client, 33.7, 2.8, date(2025, 6, 1), date(2025, 6, 3), "ecmwf_ifs025", [1, 2]
        )
    assert len(frame) == 3 * 24 * 2
    assert set(frame["lead_day"]) == {1, 2}
    assert str(frame["valid_time"].dt.tz) == "UTC"
    assert {"ghi", "dni", "dhi", "temp_air", "cloud_cover", "wind_speed"} <= set(frame.columns)
    assert elevation is not None


def test_model_suffixed_variable_names_are_accepted():
    payload = {"hourly": {"time": ["2025-01-01T00:00"], "shortwave_radiation_ecmwf_ifs025": [0.0]}}
    frame = hourly_frame(payload, {"shortwave_radiation": "ghi"}, model="ecmwf_ifs025")
    assert frame["ghi"].tolist() == [0.0]


def test_responses_are_cached_on_disk(tmp_path):
    with OpenMeteoClient(
        transport=fake_transport(), requests_per_minute=10_000, raw_dir=tmp_path
    ) as client:
        fetch_satellite(client, 30.0, 3.0, date(2025, 1, 1), date(2025, 1, 2), "eumetsat_sarah3")
        fetch_satellite(client, 30.0, 3.0, date(2025, 1, 1), date(2025, 1, 2), "eumetsat_sarah3")
        assert client.calls == 1


def test_a_rejected_request_raises_a_clear_error():
    transport = httpx.MockTransport(
        lambda r: httpx.Response(400, json={"error": True, "reason": "bad variable"})
    )
    with (
        OpenMeteoClient(transport=transport, requests_per_minute=10_000) as client,
        pytest.raises(OpenMeteoError, match="bad variable"),
    ):
        client.get("forecast", {"latitude": 0, "longitude": 0})
