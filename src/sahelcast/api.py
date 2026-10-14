"""A small read-only REST API over the serving tables.

Run it with `sahelcast serve` and open http://127.0.0.1:8000/docs for interactive docs.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .config import Settings, load_settings
from .publish import read_serving, serving_dir


def _records(frame: pd.DataFrame) -> list[dict]:
    frame = frame.copy()
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = frame[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(
        title="SahelCast API",
        description="Calibrated day-ahead solar PV forecasts and battery decisions for Algerian plants.",
        version="0.1.0",
    )

    def forecasts(live: bool) -> pd.DataFrame:
        frame = read_serving(settings, "forecasts_live" if live else "forecasts")
        if frame.empty and live:
            frame = read_serving(settings, "forecasts")
        return frame

    @app.get("/health")
    def health() -> dict:
        fc = forecasts(live=True)
        return {
            "status": "ok",
            "synthetic_data": (serving_dir(settings) / "SYNTHETIC").exists(),
            "latest_issue_time": None if fc.empty else fc["issue_time"].max().isoformat(),
        }

    @app.get("/plants")
    def plants() -> list[dict]:
        return _records(read_serving(settings, "plants"))

    @app.get("/forecast/latest")
    def latest(
        node_id: str = Query("DZ", description="DZ, DZ/<wilaya> or DZ/<wilaya>/<plant_id>"),
        model: str = Query("lightgbm+cqr"),
    ) -> dict:
        fc = forecasts(live=True)
        fc = fc[(fc["node_id"] == node_id) & (fc["model"] == model)]
        if fc.empty:
            raise HTTPException(404, f"no forecast for node {node_id!r} and model {model!r}")
        fc = fc[fc["issue_time"] == fc["issue_time"].max()]
        return _payload(fc)

    @app.get("/forecast")
    def by_date(
        issue_date: date,
        node_id: str = "DZ",
        model: str = "lightgbm+cqr",
    ) -> dict:
        fc = forecasts(live=False)
        fc = fc[
            (fc["node_id"] == node_id)
            & (fc["model"] == model)
            & (fc["issue_time"].dt.date == issue_date)
        ]
        if fc.empty:
            raise HTTPException(404, "no forecast for that date, node and model")
        return _payload(fc)

    @app.get("/metrics")
    def metrics(
        level: str | None = Query(None, description="plant, wilaya or country"),
    ) -> list[dict]:
        frame = read_serving(settings, "metrics")
        if level:
            frame = frame[frame["level"] == level]
        return _records(frame)

    @app.get("/dispatch/summary")
    def dispatch() -> list[dict]:
        return _records(read_serving(settings, "dispatch_summary"))

    return app


def _payload(fc: pd.DataFrame) -> dict:
    fc = fc.sort_values("valid_time")
    qcols = [c for c in fc.columns if c.startswith("q") and c[1:].isdigit()]
    mw = fc[qcols].mul(fc["ac_mw"], axis=0).round(3)
    hours = pd.concat(
        [fc[["valid_time"]].reset_index(drop=True), mw.reset_index(drop=True)], axis=1
    )
    return {
        "node_id": fc["node_id"].iloc[0],
        "model": fc["model"].iloc[0],
        "issue_time": fc["issue_time"].iloc[0].isoformat(),
        "units": "MW (AC)",
        "hours": _records(hours),
    }


app = create_app  # `uvicorn sahelcast.api:app --factory`
