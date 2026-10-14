import pandas as pd
from fastapi.testclient import TestClient

from sahelcast.api import create_app
from sahelcast.publish import serving_dir


def test_api_serves_the_latest_forecast(settings):
    out = serving_dir(settings)
    out.mkdir(parents=True)
    times = pd.date_range("2025-06-02 00:00", periods=24, freq="h", tz="UTC")
    fc = pd.DataFrame(
        {
            "model": "lightgbm+cqr",
            "node_id": "DZ",
            "issue_time": pd.Timestamp("2025-06-01 12:00", tz="UTC"),
            "valid_time": times,
            "ac_mw": 100.0,
            **{f"q{q:02d}": 0.1 * (i + 1) for i, q in enumerate([5, 10, 25, 50, 75, 90, 95])},
        }
    )
    fc.to_parquet(out / "forecasts.parquet")
    pd.DataFrame(
        {"model": ["lightgbm+cqr"], "level": ["country"], "quantile_score": [0.01]}
    ).to_parquet(out / "metrics.parquet")

    client = TestClient(create_app(settings))
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/forecast/latest", params={"node_id": "DZ"}).json()
    assert body["units"] == "MW (AC)"
    assert len(body["hours"]) == 24
    assert body["hours"][0]["q50"] == 40.0
    assert client.get("/forecast/latest", params={"node_id": "nowhere"}).status_code == 404
    assert client.get("/metrics", params={"level": "country"}).json()[0]["quantile_score"] == 0.01
