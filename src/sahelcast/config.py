"""Load settings (config/settings.yaml) and the plant registry (config/plants.csv)."""

from __future__ import annotations

import math
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, model_validator

ROOT = Path(__file__).resolve().parents[2]  # the repository root


class Paths(BaseModel):
    lake_metadata: Path
    lake_data: Path
    raw: Path
    artifacts: Path
    site: Path

    def resolved(self, root: Path) -> Paths:
        """Return a copy where every relative path is anchored at `root`."""
        return Paths(**{k: (v if v.is_absolute() else root / v) for k, v in self})


class OpenMeteoSettings(BaseModel):
    nwp_model: str = "ecmwf_ifs025"
    satellite_model: str = "eumetsat_sarah3"
    reanalysis_model: str = "best_match"
    history_start: date = date(2024, 3, 1)
    history_end: date | None = None
    chunk_days: int = 31
    lead_days: list[int] = [1, 2]
    requests_per_minute: int = 30

    def end(self) -> date:
        return self.history_end or (date.today() - timedelta(days=7))


class ForecastSettings(BaseModel):
    issue_hour_utc: int = 12
    lead_day: int = 2
    obs_cutoff_hours: int = 36
    quantiles: list[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    context_days: int = 28
    models: list[str] = ["persistence", "physical", "lightgbm", "chronos2"]
    chronos_model_id: str = "amazon/chronos-2"
    device: str = "cpu"


class CalibrationSettings(BaseModel):
    window_days: int = 60
    lag_days: int = 3
    gamma: float = 0.005
    mondrian: bool = True
    report_levels: list[float] = [0.8, 0.9]


class BacktestSettings(BaseModel):
    start: date
    end: date
    retrain_every_days: int = 30


class PVSettings(BaseModel):
    dc_ac_ratio: float = 1.2
    system_losses: float = 0.14
    gamma_pdc: float = -0.004
    albedo: float = 0.3
    transposition_model: str = "haydavies"


class BatterySettings(BaseModel):
    energy_hours: float = 1.0
    power_ratio: float = 0.5
    efficiency_roundtrip: float = 0.88
    initial_soc: float = 0.5
    price: float = 1.0
    shortfall_penalty: float = 3.0
    n_scenarios: int = 30
    scenario_correlation: float = 0.8


def min_lag_days(forecast: ForecastSettings) -> int:
    """Smallest calibration lag that only uses outcomes observed by the cutoff.

    The forecast issued k days before today targets a day whose last hour (hour-ending label
    23:00) must be at or before today's observation cutoff before its errors can be used.
    """
    hours = 23 - forecast.issue_hour_utc + forecast.obs_cutoff_hours
    return 1 + math.ceil(hours / 24)


class Settings(BaseModel):
    paths: Paths
    openmeteo: OpenMeteoSettings
    forecast: ForecastSettings
    calibration: CalibrationSettings
    backtest: BacktestSettings
    pv: PVSettings
    battery: BatterySettings
    root: Path = ROOT

    @model_validator(mode="after")
    def _calibration_only_uses_known_outcomes(self) -> Settings:
        need = min_lag_days(self.forecast)
        if self.calibration.lag_days < need:
            raise ValueError(
                f"calibration.lag_days={self.calibration.lag_days} would use outcomes that are "
                f"not observed yet at issue time (obs_cutoff_hours="
                f"{self.forecast.obs_cutoff_hours}): use lag_days >= {need}"
            )
        return self

    @property
    def offline(self) -> bool:
        """SAHELCAST_OFFLINE=1 swaps the real Open-Meteo API for synthetic data."""
        return os.environ.get("SAHELCAST_OFFLINE", "0") == "1"


def load_settings(path: str | Path | None = None, root: Path | None = None) -> Settings:
    root = Path(root or os.environ.get("SAHELCAST_ROOT", ROOT))
    path = Path(path or os.environ.get("SAHELCAST_SETTINGS", root / "config" / "settings.yaml"))
    raw = yaml.safe_load(path.read_text())
    settings = Settings(**raw, root=root)
    settings.paths = settings.paths.resolved(root)
    return settings


def load_plants(path: str | Path | None = None, pv: PVSettings | None = None) -> pd.DataFrame:
    """Read the plant registry and derive DC and AC capacity in MW.

    Published capacities are sometimes DC (MWp), sometimes AC (MWac) and often just "MW".
    "MW" is treated as MWp, and the other side is derived with the DC/AC ratio.
    """
    pv = pv or PVSettings()
    path = Path(path or os.environ.get("SAHELCAST_PLANTS", ROOT / "config" / "plants.csv"))
    plants = pd.read_csv(path)
    required = {"plant_id", "wilaya", "latitude", "longitude", "capacity_mw", "capacity_basis"}
    missing = required - set(plants.columns)
    if missing:
        raise ValueError(f"plants.csv is missing columns: {sorted(missing)}")
    if plants["plant_id"].duplicated().any():
        raise ValueError("plant_id must be unique in plants.csv")
    is_ac = plants["capacity_basis"].str.upper().eq("MWAC")
    plants["ac_mw"] = plants["capacity_mw"].where(is_ac, plants["capacity_mw"] / pv.dc_ac_ratio)
    plants["dc_mw"] = plants["capacity_mw"].where(~is_ac, plants["capacity_mw"] * pv.dc_ac_ratio)
    if "country" not in plants:
        plants["country"] = "DZ"
    plants["tilt_deg"] = plants.get("tilt_deg", 30).fillna(30)
    plants["azimuth_deg"] = plants.get("azimuth_deg", 180).fillna(180)
    return plants
