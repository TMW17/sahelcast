"""Turn forecasts into battery decisions, and measure what the forecasts are worth.

The decision problem (per plant, per day), in capacity-factor units (per MW of plant):
1. At issue time the plant commits an hourly delivery schedule c[t] for tomorrow. Each
   committed MWh earns `price`; each MWh it then fails to deliver costs `shortfall_penalty`
   (the grid operator has to start gas units instead).
2. Tomorrow, the battery charges from surplus PV and discharges to cover shortfalls,
   following a simple real-time rule that cannot see the future.

Planning uses a linear program: for each scenario of tomorrow's PV, the battery is allowed
to plan ahead; the commitment must be the same in all scenarios. Execution uses the real-
time rule, so a plan that looked good only because it assumed perfect battery foresight
is penalised, as it would be in reality.

Policies compared:
- perfect_foresight                knows tomorrow's actual PV: the best possible profit
- deterministic_raw_median         plans as if the model's median were certain
- deterministic_calibrated_median  the same with the calibrated median
- stochastic_raw                   plans over scenarios sampled from the raw quantiles
- stochastic_calibrated            the same, from the conformally calibrated quantiles
Regret = perfect-foresight profit - policy profit. Lower regret = more valuable forecasts.

Without a battery the best commitment is the forecast quantile at price / penalty (the
"newsvendor" rule): with price 1 and penalty 3 that is the 33rd percentile, not the median.
So the decision depends on the quantiles being right, which is what calibration buys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import BatterySettings
from .dataset import qcol

log = logging.getLogger(__name__)

HOURS = 24


@dataclass(frozen=True)
class Battery:
    energy: float  # MWh per MW of plant AC capacity
    power: float  # MW per MW of plant AC capacity
    efficiency: float  # one-way efficiency
    soc0: float  # state of charge at midnight, fraction of energy capacity

    @classmethod
    def from_settings(cls, b: BatterySettings) -> Battery:
        return cls(
            b.energy_hours, b.power_ratio, float(np.sqrt(b.efficiency_roundtrip)), b.initial_soc
        )


def _solver() -> str:
    return "HIGHS" if "HIGHS" in cp.installed_solvers() else "CLARABEL"


class DispatchModel:
    """Linear programs built once with parameters, then re-solved quickly for every day."""

    def __init__(self, battery: Battery, price: float, penalty: float, n_scenarios: int) -> None:
        self.battery, self.price, self.penalty = battery, price, penalty
        self.solver = _solver()
        self.plan = {n: self._planning_problem(n) for n in {1, n_scenarios}}

    def _operations(self, pv, commit, n: int):
        """Battery and energy-balance constraints for n scenarios (rows)."""
        b = self.battery
        e_max, p_max = b.energy, b.power
        charge = cp.Variable((n, HOURS), nonneg=True)
        discharge = cp.Variable((n, HOURS), nonneg=True)
        soc = cp.Variable((n, HOURS + 1), nonneg=True)
        short = cp.Variable((n, HOURS), nonneg=True)
        spill = cp.Variable((n, HOURS), nonneg=True)
        commit_rows = np.ones((n, 1)) @ cp.reshape(commit, (1, HOURS), order="C")
        cons = [
            soc[:, 0] == b.soc0 * e_max,
            soc[:, 1:] == soc[:, :-1] + b.efficiency * charge - discharge / b.efficiency,
            soc <= e_max,
            soc[:, HOURS] >= b.soc0 * e_max,  # end the day with the energy you started with
            charge <= p_max,
            discharge <= p_max,
            spill <= pv,
            pv - spill - charge + discharge + short == commit_rows,
        ]
        return short, cons

    def _planning_problem(self, n: int):
        pv = cp.Parameter((n, HOURS), nonneg=True)
        commit = cp.Variable(HOURS, nonneg=True)
        short, cons = self._operations(pv, commit, n)
        cons.append(commit <= 1.0)  # cannot deliver more than the grid connection allows
        objective = cp.Maximize(self.price * cp.sum(commit) - self.penalty * cp.sum(short) / n)
        return cp.Problem(objective, cons), pv, commit

    def commit(self, scenarios: np.ndarray) -> np.ndarray:
        problem, pv, commit = self.plan[scenarios.shape[0]]
        pv.value = np.clip(scenarios, 0, None)
        problem.solve(solver=self.solver)
        return np.clip(commit.value, 0, 1)

    def realize(self, commitment: np.ndarray, actual: np.ndarray) -> dict[str, float]:
        """What really happens tomorrow: the battery follows a simple hour-by-hour rule.

        Each hour, surplus PV charges the battery (the rest is spilled) and any deficit is
        covered from the battery as far as it can; what remains is a shortfall. The rule
        cannot see the future, like a real controller. Energy left in (or taken from) the
        battery at midnight is valued at the energy price.
        """
        b = self.battery
        soc = b.soc0 * b.energy
        shortfall = spilled = 0.0
        for pv_t, c_t in zip(np.clip(actual, 0, None), np.clip(commitment, 0, None), strict=True):
            surplus = pv_t - c_t
            if surplus >= 0:
                charge = min(surplus, b.power, (b.energy - soc) / b.efficiency)
                soc += charge * b.efficiency
                spilled += surplus - charge
            else:
                discharge = min(-surplus, b.power, soc * b.efficiency)
                soc -= discharge / b.efficiency
                shortfall += -surplus - discharge
        delivered = float(np.clip(commitment, 0, None).sum())
        stored_change = soc - b.soc0 * b.energy
        return {
            "committed": delivered,
            "shortfall": shortfall,
            "spilled": spilled,
            "profit": self.price * (delivered + stored_change) - self.penalty * shortfall,
        }


def scenarios_from_quantiles(
    values: np.ndarray, taus: list[float], n: int, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample n PV paths (n x 24) whose hourly distribution matches the forecast quantiles.

    Hours are linked by an AR(1) Gaussian copula (correlation `rho` between consecutive
    hours): a cloudy morning makes a cloudy afternoon more likely.
    """
    q = np.sort(values, axis=1)
    low = np.clip(q[:, 0] - (q[:, 1] - q[:, 0]), 0, None)
    high = np.clip(q[:, -1] + (q[:, -1] - q[:, -2]), None, 1.0)
    grid = np.column_stack([low, q, high])
    tau_grid = np.array([0.0, *taus, 1.0])
    z = np.empty((n, values.shape[0]))
    z[:, 0] = rng.standard_normal(n)
    for t in range(1, values.shape[0]):
        z[:, t] = rho * z[:, t - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    u = norm.cdf(z)
    return np.column_stack([np.interp(u[:, t], tau_grid, grid[t]) for t in range(values.shape[0])])


def estimate_hourly_correlation(forecasts: pd.DataFrame, quantiles: list[float]) -> float:
    """How strongly an error at one hour carries over to the next (for the scenario copula).

    Each observation is turned into its probability under the forecast (its PIT value), then
    into a normal score; the lag-1 correlation of those scores within a day is returned.
    """
    qcols = [qcol(q) for q in quantiles]
    fc = forecasts[(forecasts["cf_clearsky"] > 0.05) & forecasts["y"].notna()]
    fc = fc.sort_values(["node_id", "issue_time", "valid_time"])
    taus = np.array(quantiles)
    q = fc[qcols].to_numpy()
    y = fc["y"].to_numpy()
    pit = np.array(
        [np.interp(v, row, taus, left=0.01, right=0.99) for v, row in zip(y, q, strict=True)]
    )
    z = pd.Series(norm.ppf(np.clip(pit, 0.01, 0.99)), index=fc.index)
    same_day = fc.groupby(["node_id", "issue_time"]).cumcount() > 0
    lagged = z.groupby([fc["node_id"], fc["issue_time"]]).shift(1)
    ok = same_day & lagged.notna()
    if ok.sum() < 100:
        return 0.8
    return float(np.clip(np.corrcoef(z[ok], lagged[ok])[0, 1], 0.0, 0.99))


def run_dispatch_study(
    forecasts: pd.DataFrame,
    model: str,
    battery_settings: BatterySettings,
    quantiles: list[float],
    days: int | None = 60,
    seed: int = 0,
) -> pd.DataFrame:
    """Profit and regret of each policy for every plant and day of the last `days` days.

    The scenario correlation is estimated on the calibrated forecasts issued before that
    period, so the study never uses information from the days it evaluates.
    """
    b = battery_settings
    dm = DispatchModel(Battery.from_settings(b), b.price, b.shortfall_penalty, b.n_scenarios)
    rng = np.random.default_rng(seed)
    qcols = [qcol(q) for q in quantiles]
    fc = forecasts[(forecasts["level"] == "plant") & forecasts["in_evaluation"]]
    raw = fc[fc["model"] == model]
    cal = fc[fc["model"] == f"{model}+cqr"]
    issue_days = sorted(raw["issue_time"].unique())
    if days:
        issue_days = issue_days[-days:]
    earlier = cal[cal["issue_time"] < issue_days[0]]
    rho = (
        estimate_hourly_correlation(earlier, quantiles) if len(earlier) else b.scenario_correlation
    )
    log.info("dispatch: scenario hour-to-hour correlation %.2f", rho)
    rows = []
    for node, raw_node in raw.groupby("node_id"):
        cal_node = cal[cal["node_id"] == node].set_index(["issue_time", "valid_time"])
        for issue_time in issue_days:
            day_raw = raw_node[raw_node["issue_time"] == issue_time].sort_values("valid_time")
            if len(day_raw) != HOURS or day_raw["y"].isna().any():
                continue  # skip days with missing observations
            actual = day_raw["y"].to_numpy()
            q_raw = day_raw[qcols].to_numpy()
            q_cal = cal_node.loc[issue_time].loc[day_raw["valid_time"], qcols].to_numpy()
            median = quantiles.index(0.5)
            plans = {
                "perfect_foresight": actual[None, :],
                "deterministic_raw_median": q_raw[:, median][None, :],
                "deterministic_calibrated_median": q_cal[:, median][None, :],
                "stochastic_raw": scenarios_from_quantiles(
                    q_raw, quantiles, b.n_scenarios, rho, rng
                ),
                "stochastic_calibrated": scenarios_from_quantiles(
                    q_cal, quantiles, b.n_scenarios, rho, rng
                ),
            }
            for policy, scenarios in plans.items():
                result = dm.realize(dm.commit(scenarios), actual)
                rows.append({"node_id": node, "issue_time": issue_time, "policy": policy, **result})
    out = pd.DataFrame(rows)
    best = out[out["policy"] == "perfect_foresight"].set_index(["node_id", "issue_time"])["profit"]
    out["regret"] = out.set_index(["node_id", "issue_time"]).index.map(best) - out["profit"]
    out["model"] = model
    return out


def summarize_dispatch(results: pd.DataFrame) -> pd.DataFrame:
    """Average daily profit and regret per MW of plant, and the share of the ideal profit."""
    summary = results.groupby("policy").agg(
        plant_days=("profit", "size"),
        profit_per_mw_day=("profit", "mean"),
        regret_per_mw_day=("regret", "mean"),
        shortfall_mwh_per_mw_day=("shortfall", "mean"),
    )
    ideal = summary.loc["perfect_foresight", "profit_per_mw_day"]
    summary["share_of_ideal_profit"] = summary["profit_per_mw_day"] / ideal
    order = [
        "perfect_foresight",
        "stochastic_calibrated",
        "stochastic_raw",
        "deterministic_calibrated_median",
        "deterministic_raw_median",
    ]
    return summary.reindex([p for p in order if p in summary.index]).reset_index()
