"""Make plant, wilaya and national forecasts add up (hierarchical reconciliation).

Every node is forecast separately, so the national median is usually not the sum of the
plant medians. Reconciliation adjusts all levels together so they are coherent:
- BottomUp                sum the plant forecasts (ignores the aggregate forecasts)
- MinTrace (wls_struct)   weights each level by how many plants it contains
- MinTrace (mint_shrink)  weights by the error covariance learned on earlier days

This is done on medians in MW (quantiles do not add up). Intervals at each level come from
that level's own calibrated forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace
from hierarchicalforecast.utils import aggregate

SPEC = [["country"], ["country", "wilaya"], ["country", "wilaya", "plant"]]


def hierarchy(plants: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Summing matrix S (which plants make up each node) and the node names per level."""
    frame = pd.DataFrame(
        {
            "country": plants["country"].to_numpy() if "country" in plants else "DZ",
            "wilaya": plants["wilaya"].to_numpy(),
            "plant": plants["plant_id"].to_numpy(),
            "ds": pd.Timestamp("2025-01-01"),
            "y": 0.0,
        }
    )
    _, s_df, tags = aggregate(frame, SPEC)
    return s_df, tags


def reconcile_medians(
    fc: pd.DataFrame, plants: pd.DataFrame, insample_days: int = 60
) -> pd.DataFrame:
    """Reconcile one model's medians. The first `insample_days` issue days only teach MinT
    the error covariance; reconciled forecasts are returned for the days after that."""
    s_df, tags = hierarchy(plants)
    fc = fc.assign(
        base=fc["q50"] * fc["ac_mw"],
        y_mw=fc["y"] * fc["ac_mw"],
        ds=fc["valid_time"].dt.tz_convert("UTC").dt.tz_localize(None),
    ).rename(columns={"node_id": "unique_id"})
    split = fc["issue_time"].min() + pd.Timedelta(days=insample_days)
    insample = fc[fc["issue_time"] < split]
    outsample = fc[fc["issue_time"] >= split]

    # MinT needs complete in-sample residuals: keep only hours observed at every node.
    wide = insample.pivot_table(index="ds", columns="unique_id", values="y_mw")
    complete = wide.dropna().index
    y_df = insample[insample["ds"].isin(complete)][["unique_id", "ds", "y_mw", "base"]]
    y_df = y_df.rename(columns={"y_mw": "y"})

    hrec = HierarchicalReconciliation(
        reconcilers=[BottomUp(), MinTrace(method="wls_struct"), MinTrace(method="mint_shrink")]
    )
    rec = hrec.reconcile(
        Y_hat_df=outsample[["unique_id", "ds", "base"]], Y_df=y_df, S_df=s_df, tags=tags
    )
    rec = rec.merge(
        outsample[["unique_id", "ds", "y_mw", "cf_clearsky", "level"]], on=["unique_id", "ds"]
    )
    return rec.rename(columns={"unique_id": "node_id"})


def evaluate_reconciliation(rec: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error (MW, daylight hours) of base and reconciled medians by level."""
    rec = rec[(rec["cf_clearsky"] > 0.01) & rec["y_mw"].notna()]
    methods = [c for c in rec.columns if c.startswith("base")]
    rows = []
    for level, g in rec.groupby("level"):
        row = {"level": level}
        for m in methods:
            row[m.replace("base/", "") if "/" in m else "base (incoherent)"] = float(
                np.mean(np.abs(g[m] - g["y_mw"]))
            )
        rows.append(row)
    return pd.DataFrame(rows)
