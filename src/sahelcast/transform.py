"""Run the dbt project (silver and gold layers) against the lake.

dbt runs in a separate process. DuckDB lets only one open handle per database file, and
dbt keeps its connection open until its process ends, so running it in-process would
block the next pipeline step from opening the lake.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import Settings


def dbt_env(settings: Settings) -> dict[str, str]:
    paths = settings.paths
    return {
        "SAHELCAST_LAKE_METADATA": str(paths.lake_metadata),
        "SAHELCAST_LAKE_DATA": str(paths.lake_data).rstrip("/") + "/",
        "SAHELCAST_DBT_DB": str(Path(paths.lake_metadata).parent.parent / "dbt.duckdb"),
    }


def dbt_command(settings: Settings, args: list[str] | None = None) -> list[str]:
    project_dir = settings.root / "dbt"
    variables = {
        "nwp_model": settings.openmeteo.nwp_model,
        "satellite_model": settings.openmeteo.satellite_model,
        "reanalysis_model": settings.openmeteo.reanalysis_model,
        "lead_days": settings.openmeteo.lead_days,
    }
    executable = shutil.which("dbt", path=str(Path(sys.executable).parent)) or "dbt"
    return [
        executable,
        *(args or ["build"]),
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(project_dir),
        "--vars",
        json.dumps(variables),
    ]


def run_dbt(settings: Settings, args: list[str] | None = None) -> bool:
    """Run a dbt command (default `dbt build`: models plus their tests). True if all passed."""
    Path(settings.paths.lake_metadata).parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **dbt_env(settings)}
    result = subprocess.run(dbt_command(settings, args), env=env, cwd=settings.root / "dbt")
    return result.returncode == 0
