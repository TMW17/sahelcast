"""Chronos-2 (Amazon, Apache-2.0), a pretrained time-series foundation model, used zero-shot.

For each node and issue day the model sees:
- the past `context_days` of observed capacity factor (the target)
- past and future covariates: the NWP-based physical forecast and the clear-sky capacity factor
It returns quantiles for every hour from the observation cutoff to the end of the target day;
only the 24 target hours are kept.

The first call downloads the weights (about 0.5 GB for amazon/chronos-2) from Hugging Face.
On an Apple Silicon Mac, set `device: mps` in settings.yaml to use the GPU.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .baselines import finalize
from .config import Settings
from .dataset import Issue, Panel, qcol

log = logging.getLogger(__name__)


class Chronos2Forecaster:
    name = "chronos2"

    def __init__(
        self, model_id: str = "amazon/chronos-2", device: str = "cpu", pipeline=None
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._pipeline = pipeline  # tests inject a tiny model here

    @property
    def pipeline(self):
        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            log.info("loading %s on %s", self.model_id, self.device)
            self._pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device)
        return self._pipeline

    def forecast(
        self, panel: Panel, issues: list[Issue], settings: Settings, block_days: int = 30
    ) -> pd.DataFrame:
        fs = settings.forecast
        lead = fs.lead_day
        covariates = [f"cf_nwp{lead}", "cf_clearsky"]
        out = []
        for start in range(0, len(issues), block_days):
            block = issues[start : start + block_days]
            contexts, futures, targets = [], [], {}
            for issue in block:
                for node in panel.nodes:
                    hist = panel.history(node, issue.cutoff, fs.context_days)
                    fut = panel.window(node, issue.cutoff + pd.Timedelta(hours=1), issue.target_end)
                    if hist["target_cf"].notna().sum() < 24 * 7 or fut.empty:
                        continue
                    item = f"{node}|{issue.day.isoformat()}"
                    ctx = hist[["target_cf", *covariates]].copy()
                    ctx[covariates] = ctx[covariates].ffill().bfill().fillna(0.0)
                    contexts.append(ctx.reset_index().assign(item_id=item))
                    f = (
                        fut[covariates]
                        .ffill()
                        .bfill()
                        .fillna(0.0)
                        .reset_index()
                        .assign(item_id=item)
                    )
                    futures.append(f)
                    targets[item] = (node, issue, fut.loc[issue.target_start : issue.target_end])
            if not contexts:
                continue
            context_df = self._naive(pd.concat(contexts, ignore_index=True))
            future_df = self._naive(pd.concat(futures, ignore_index=True))
            horizon = int(future_df.groupby("item_id").size().max())
            pred = self.pipeline.predict_df(
                context_df,
                future_df=future_df,
                id_column="item_id",
                timestamp_column="valid_time",
                target="target_cf",
                prediction_length=horizon,
                quantile_levels=fs.quantiles,
            )
            pred["valid_time"] = pd.to_datetime(pred["valid_time"]).dt.tz_localize("UTC")
            for item, g in pred.groupby("item_id"):
                node, issue, target = targets[item]
                g = g.set_index("valid_time").reindex(target.index)
                values = np.column_stack([g[str(q)].to_numpy(dtype=float) for q in fs.quantiles])
                cs = target["cf_clearsky"].to_numpy()
                frame = pd.DataFrame(finalize(values, cs), columns=[qcol(q) for q in fs.quantiles])
                frame.insert(0, "valid_time", target.index)
                frame.insert(0, "issue_time", issue.time)
                frame.insert(0, "node_id", node)
                frame.insert(0, "model", self.name)
                out.append(frame)
            log.info("chronos2: block starting %s done", block[0].day)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    @staticmethod
    def _naive(frame: pd.DataFrame) -> pd.DataFrame:
        """Chronos works with timezone-naive timestamps; everything here is UTC."""
        frame["valid_time"] = (
            pd.to_datetime(frame["valid_time"]).dt.tz_convert("UTC").dt.tz_localize(None)
        )
        return frame.sort_values(["item_id", "valid_time"]).reset_index(drop=True)
