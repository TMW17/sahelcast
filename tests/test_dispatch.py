import numpy as np
import pytest

from sahelcast.dispatch import Battery, DispatchModel, scenarios_from_quantiles

QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
PV = np.clip(np.sin((np.arange(24) - 6) / 12 * np.pi), 0, None) * 0.8


def model(n=20):
    return DispatchModel(Battery(energy=1.0, power=0.5, efficiency=0.94, soc0=0.5), 1.0, 3.0, n)


def test_perfect_foresight_has_no_shortfall_and_earns_all_energy():
    dm = model()
    result = dm.realize(dm.commit(PV[None, :]), PV)
    assert result["shortfall"] == pytest.approx(0.0, abs=1e-6)
    assert result["profit"] == pytest.approx(PV.sum(), rel=1e-4)


def test_battery_covers_a_deficit_up_to_its_power_and_energy():
    dm = model()
    commitment = np.zeros(24)
    commitment[20] = 0.4  # evening: no sun
    result = dm.realize(commitment, np.zeros(24))
    assert result["shortfall"] == pytest.approx(0.0, abs=1e-9)  # 0.4 MWh <= power 0.5 and SOC 0.5
    commitment[21] = 0.4
    result = dm.realize(commitment, np.zeros(24))
    assert result["shortfall"] > 0.2  # the battery ran out


def test_scenarios_reproduce_the_forecast_quantiles():
    q = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), (24, 1))
    sc = scenarios_from_quantiles(q, QUANTILES, 20_000, 0.8, np.random.default_rng(0))
    assert np.allclose(np.quantile(sc[:, 5], QUANTILES), q[5], atol=0.01)


def test_higher_penalty_makes_the_plan_more_cautious():
    rng = np.random.default_rng(0)
    q = np.column_stack([PV * f for f in (0.5, 0.6, 0.8, 1.0, 1.1, 1.2, 1.25)])
    scenarios = scenarios_from_quantiles(q, QUANTILES, 30, 0.8, rng)
    cheap = DispatchModel(Battery(1.0, 0.5, 0.94, 0.5), 1.0, 1.5, 30).commit(scenarios).sum()
    dear = DispatchModel(Battery(1.0, 0.5, 0.94, 0.5), 1.0, 6.0, 30).commit(scenarios).sum()
    assert dear < cheap
