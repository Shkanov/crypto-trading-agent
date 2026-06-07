"""Expanding-origin, calendar-anchored, embargoed walk-forward.

The directional probe that found 0.564 used ONE chronological train/test split.
That is the weakest possible evidence — a single draw. Here every prediction is
strictly out-of-sample and forward-only, and we make many of them:

  * partition the pooled samples' calendar into `n_blocks` contiguous time blocks,
  * walk the last `oos_blocks` of them forward (expanding origin): for OOS block
    b, TRAIN on every sample whose target is realised strictly before b's start
    MINUS an embargo of `lookback` blocks, then PREDICT block b,
  * the embargo drops train samples whose feature window (L lagged returns)
    reaches into the test period — the AR analogue of AFML purging.

Pooling across coins gives honest n; the calendar anchoring keeps it PIT (a
prediction for week t only ever sees returns realised < t, across ALL coins).
Per fold we fit on the TRAIN slice, StandardScaler-ed on TRAIN only (PIT scaling
— no test leakage into the mean/var), and score the held-out block. Returns one
tidy row per (model, coin, target-week) OOS prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from research.price_predict.models import LEARNED, MODELS, make_model
from research.price_predict.series import WindowSpec, feature_cols


@dataclass(frozen=True)
class WalkForwardConfig:
    n_blocks: int = 10            # calendar partitions over the full sample span
    oos_blocks: int = 6          # how many trailing blocks to score (expanding)
    embargo_steps: int = 15      # train target must end this many blocks before test
    min_train: int = 200         # skip a fold without enough pooled train samples
    seed: int = 42


def _time_blocks(t: pd.Series, n_blocks: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split the calendar span of `t` into n_blocks contiguous [start, end) ranges
    by equal time width. Returns (start, end) pairs in chronological order."""
    t0, t1 = t.min(), t.max()
    # Preserve tz (samples are UTC); a tz-naive edge can't be compared to them.
    edges = pd.to_datetime(np.linspace(t0.value, t1.value, n_blocks + 1), utc=True)
    # make the last edge inclusive of the final timestamp
    return [(edges[i], edges[i + 1]) for i in range(n_blocks)]


def run_walkforward(samples: pd.DataFrame, spec: WindowSpec,
                    cfg: WalkForwardConfig | None = None,
                    models: tuple[str, ...] = MODELS) -> pd.DataFrame:
    """Pooled walk-forward → OOS predictions.

    `samples`: output of series.build_pooled_samples (coin, t, f0..f{L-1}, y).
    Returns columns: model, coin, t, pred, y. Empty if no fold has enough data.
    """
    cfg = cfg or WalkForwardConfig()
    fcols = feature_cols(spec)
    samples = samples.sort_values("t").reset_index(drop=True)
    blocks = _time_blocks(samples["t"], cfg.n_blocks)
    step = pd.Timedelta(days=spec.step_days)

    out_rows: list[pd.DataFrame] = []
    for b_start, b_end in blocks[-cfg.oos_blocks:]:
        test_mask = (samples["t"] >= b_start) & (samples["t"] < b_end)
        # include the final timestamp in the very last block
        if b_end == blocks[-1][1]:
            test_mask = (samples["t"] >= b_start) & (samples["t"] <= b_end)
        test = samples[test_mask]
        if test.empty:
            continue
        embargo_cut = b_start - cfg.embargo_steps * step
        train = samples[samples["t"] < embargo_cut]
        if len(train) < cfg.min_train:
            continue

        Xtr_raw = train[fcols].to_numpy()
        scaler = StandardScaler().fit(Xtr_raw)
        Xtr = scaler.transform(Xtr_raw)
        ytr = train["y"].to_numpy()
        Xte = scaler.transform(test[fcols].to_numpy())

        for name in models:
            m = make_model(name, cfg.seed)
            if name in LEARNED:
                m.fit(Xtr, ytr)
            pred = np.asarray(m.predict(Xte), dtype=float).reshape(-1)
            out_rows.append(pd.DataFrame({
                "model": name,
                "coin": test["coin"].to_numpy(),
                "t": test["t"].to_numpy(),
                "pred": pred,
                "y": test["y"].to_numpy(),
            }))

    if not out_rows:
        return pd.DataFrame(columns=["model", "coin", "t", "pred", "y"])
    return pd.concat(out_rows, ignore_index=True)
