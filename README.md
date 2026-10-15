# SahelCast

[![ci](https://github.com/TMW17/sahelcast/actions/workflows/ci.yml/badge.svg)](https://github.com/TMW17/sahelcast/actions/workflows/ci.yml)

**Live results page:** https://tmw17.github.io/sahelcast/

**Day-ahead solar power forecasts with calibrated uncertainty for Algerian PV plants, and what they are worth for battery scheduling.**

SahelCast collects weather forecasts (as they were issued) and satellite irradiance for Algeria's
utility solar plants into a DuckLake lakehouse, turns them into PV power with pvlib, forecasts
tomorrow's output as probability ranges (persistence, physics, LightGBM, Chronos-2), calibrates
those ranges with conformal prediction, makes plant, wilaya and national forecasts add up, and
uses them to plan a battery. Success is measured in money (regret), not only in forecast error.

**Results** (backtest of every day of 2025, 10 Algerian plants, national level; "observed" power is simulated from satellite irradiance):
- LightGBM forecasts: quantile score 0.0087, **37% better than persistence**; its 80% and 90% ranges contained the outcome 81% and 91% of the time.
- Conformal calibration fixed the overconfident persistence baseline (80% range coverage 56% → 78%) but did not improve LightGBM, which was already well calibrated.
- Battery planning over forecast scenarios lost 10% less profit than planning on the median (regret 0.36 vs 0.40 per MW-day) and reached 93% of the perfect-foresight profit (600 plant-days).
- MinT reconciliation lowered the national median error by 2% (MAE 9.92 → 9.73 MW).

## Architecture

```text
Open-Meteo APIs ──► bronze (DuckLake) ──► dbt: silver, gold ──► pvlib: gold.pv_hourly
 (previous runs,        raw, typed            cleaned, tested          PV power, clear-sky
  satellite, ERA5)                                                        │
                                                                          ▼
 website ◄── serving tables ◄── dispatch (CVXPY) ◄── reconcile ◄── calibrate (CQR+ACI) ◄── models
 (static)      (Parquet)           battery plans       (MinT)        per-quantile, Mondrian     persistence
 FastAPI                                                                                        physical
 Dagster: daily forecast at 12:15 UTC, weekly evaluation                                        LightGBM
                                                                                                 Chronos-2
```

## Quick start (macOS, Linux or WSL)

```bash
brew install uv libomp          # macOS (Linux: curl -LsSf https://astral.sh/uv/install.sh | sh)
git clone https://github.com/TMW17/sahelcast && cd sahelcast
make setup                      # installs everything with uv
make demo                       # full pipeline offline on synthetic data (about 4 minutes)
open data/demo/site/index.html
```

With real data:

```bash
make ingest        # history from Open-Meteo (free, no key; first run 30-90 minutes)
make pipeline      # dbt, pvlib, backtest, calibration, reconciliation, dispatch, website
make serve         # REST API: http://127.0.0.1:8000/docs
make dagster       # schedules: http://127.0.0.1:3000
```

The full step-by-step build guide is in [BUILD_GUIDE.md](BUILD_GUIDE.md).

## What is where

| Path | What it does |
|---|---|
| `config/plants.csv` | Plant registry (public sources, see the `source` column) |
| `config/settings.yaml` | Every setting: models, schedule, calibration, battery |
| `src/sahelcast/openmeteo.py` | API client: rate limit, retries, raw-response cache |
| `src/sahelcast/lake.py` | DuckLake lakehouse, idempotent upserts, snapshots |
| `dbt/` | Silver and gold models plus data tests |
| `src/sahelcast/physics.py` | pvlib: sun position, clear sky, PV power |
| `src/sahelcast/dataset.py` | Plant/wilaya/national panel and the leak-free schedule |
| `src/sahelcast/baselines.py`, `chronos_model.py` | The four forecasting models |
| `src/sahelcast/conformal.py` | Per-quantile conformal calibration with ACI |
| `src/sahelcast/reconcile.py` | Hierarchical reconciliation (BottomUp, MinT) |
| `src/sahelcast/dispatch.py` | Battery planning LP, scenarios, regret |
| `src/sahelcast/live.py` | Tomorrow's forecast, every day |
| `src/sahelcast/api.py`, `report.py` | REST API and static website |
| `src/sahelcast/orchestration/` | Dagster assets, jobs and schedules |
| `tests/` | Unit tests, leakage tests, offline end-to-end test |

## Data sources and credits

- Weather forecasts, satellite irradiance and reanalysis: [Open-Meteo.com](https://open-meteo.com/),
  CC BY 4.0 (free for non-commercial use). Open-Meteo redistributes ECMWF, EUMETSAT and DWD products.
- Satellite irradiance: SARAH-3 climate data record, EUMETSAT CM SAF
  ([Pfeifroth et al., 2024](https://essd.copernicus.org/articles/16/5243/2024/)).
- Plant list: Global Energy Monitor wiki and press reports (see `config/plants.csv`).
- Chronos-2: Amazon, Apache-2.0.

## Limitations

- Output data of Algerian plants is not public: "observed" power is simulated from satellite
  irradiance with pvlib. The guide shows how to validate the method on real plants (NREL PVDAQ)
  and real grid data (Elia).
- Plant tilt and azimuth are assumed (30°, south). Coordinates marked `approximate` are town-level.
- SARAH-3 satellite data are published about two days late, so every forecast only uses
  observations up to 36 hours before it is issued (the backtest follows the same rule).
- The battery economics use illustrative prices, not Algerian tariffs.
