# Every command runs inside the project's environment through `uv run`.
.PHONY: setup demo ingest ingest-live transform physics backtest reconcile dispatch forecast \
        publish report serve pipeline daily dagster test test-fast lint snapshots clean

setup:            ## install everything (including Chronos-2) and the git hooks
	uv sync --all-extras
	uv run pre-commit install

demo:             ## the whole pipeline offline, on synthetic data (data/demo)
	uv run sahelcast demo

ingest:           ## download history from Open-Meteo (first run: 30-90 minutes)
	uv run sahelcast ingest

ingest-live:      ## download the latest forecast
	uv run sahelcast ingest-live

transform:        ## dbt build: silver and gold tables and their data tests
	uv run sahelcast transform

physics:          ## PV power with pvlib
	uv run sahelcast physics

backtest:         ## rolling-origin backtest, calibration, scores
	uv run sahelcast backtest

reconcile:
	uv run sahelcast reconcile

dispatch:         ## battery decisions and the sensitivity grid
	uv run sahelcast dispatch --grid

forecast:         ## tomorrow's forecast
	uv run sahelcast forecast

publish:
	uv run sahelcast publish

report:           ## build site/index.html
	uv run sahelcast report

serve:            ## REST API on http://127.0.0.1:8000/docs
	uv run sahelcast serve

pipeline:         ## every step after ingestion, in order
	uv run sahelcast pipeline --skip-ingest

daily:            ## what the daily schedule does
	uv run sahelcast ingest --start $$(date -v-10d +%F 2>/dev/null || date -d '10 days ago' +%F) --end $$(date +%F)
	uv run sahelcast ingest-live
	uv run sahelcast transform
	uv run sahelcast physics
	uv run sahelcast forecast
	uv run sahelcast publish
	uv run sahelcast report

dagster:          ## Dagster UI on http://127.0.0.1:3000
	uv run dagster dev

test:             ## all tests, including the end-to-end pipeline (about 2 minutes)
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

snapshots:        ## the lake's version history
	uv run sahelcast snapshots

clean:            ## delete generated data (keeps data/raw so nothing is downloaded twice)
	rm -rf data/lake data/artifacts data/dbt.duckdb data/demo site dbt/target dbt/logs
