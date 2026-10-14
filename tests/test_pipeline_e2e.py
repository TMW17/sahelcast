"""The whole pipeline on synthetic data: ingest -> dbt -> pvlib -> backtest -> decisions -> site."""

from argparse import Namespace
from datetime import date, timedelta

import pytest

from sahelcast import cli
from sahelcast.lake import Lake


@pytest.mark.slow
def test_offline_pipeline_end_to_end(settings):
    end = date.today() - timedelta(days=7)
    settings.backtest.start = end - timedelta(days=70)
    settings.backtest.end = end - timedelta(days=1)
    cli.cmd_ingest(
        settings,
        Namespace(
            start=end - timedelta(days=120), end=end, sources=["nwp", "satellite", "reanalysis"]
        ),
    )
    cli.cmd_transform(settings, Namespace(dbt_args=None))
    cli.cmd_physics(settings, Namespace())
    cli.cmd_backtest(
        settings, Namespace(models=["persistence", "physical", "lightgbm"], start=None, end=None)
    )
    cli.cmd_reconcile(settings, Namespace(model=None))
    cli.cmd_dispatch(settings, Namespace(model=None, days=3, grid=False, grid_days=3))
    cli.cmd_publish(settings, Namespace())
    cli.cmd_report(settings, Namespace())

    with Lake.from_settings(settings, read_only=True) as lake:
        metrics = lake.read("gold.metrics")
        assert {"persistence+cqr", "physical+cqr", "lightgbm+cqr"} <= set(metrics["model"])
        assert lake.table_exists("gold.dispatch_summary")
    page = (settings.paths.site / "index.html").read_text()
    assert "Synthetic data" in page and "Plotly.react" in page
