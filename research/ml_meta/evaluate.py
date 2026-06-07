"""Honest OOS evaluation: meta-filtered vs RAW primary, over span-purged CPCV.

The only baseline that matters is the raw primary (take every signal). For each
combinatorial split we train the meta-model on the purged training events and
score the held-out events, then compare the meta-filtered subset's OOS economics
to the raw set's — ON THE SAME held-out events. Aggregating across splits gives
a distribution (not a single lucky number), which is what we report.

`ret_net` is the per-trade directional return AFTER round-trip cost (from the
cost-aware triple-barrier label). A trade's "win" = ret_net > 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.ml_meta.cv import PurgedSplit
from research.ml_meta.features import FEATURE_COLS
from research.ml_meta.model import predict_win_proba, train


def _econ(returns: np.ndarray) -> dict:
    """Per-trade economics for a set of taken trades."""
    n = len(returns)
    if n == 0:
        return dict(n=0, mean=0.0, total=0.0, win=0.0, ir=0.0)
    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if n > 1 else 0.0
    return dict(
        n=n, mean=mean, total=float(returns.sum()),
        win=float((returns > 0).mean()),
        ir=(mean / std) if std > 0 else 0.0,     # per-trade information ratio
    )


@dataclass
class FoldResult:
    test_folds: tuple
    raw: dict
    meta: dict
    n_take: int
    n_test: int


def evaluate_cpcv(ds: pd.DataFrame, splits: list[PurgedSplit],
                  *, params: dict | None = None,
                  threshold: float = 0.5,
                  feature_cols: list[str] | None = None,
                  min_train: int = 50, min_test: int = 8) -> list[FoldResult]:
    """Run the meta-model through every purged split and compare to raw.
    `feature_cols` defaults to the mean-rev FEATURE_COLS; pass the funding
    primary's column list for Phase 2."""
    X = ds[feature_cols or FEATURE_COLS]
    y = ds["y"]
    w = ds["w"]
    r = ds["ret_net"].values
    out: list[FoldResult] = []
    for s in splits:
        tr, te = s.train_idx, s.test_idx
        if len(tr) < min_train or len(te) < min_test:
            continue
        if y.iloc[tr].nunique() < 2:
            continue                      # need both classes to train
        m = train(X.iloc[tr], y.iloc[tr], w.iloc[tr], params)
        p = predict_win_proba(m, X.iloc[te])
        take = p >= threshold
        out.append(FoldResult(
            test_folds=s.test_folds,
            raw=_econ(r[te]),
            meta=_econ(r[te][take]),
            n_take=int(take.sum()),
            n_test=len(te),
        ))
    return out


def summarize(results: list[FoldResult]) -> dict:
    """Aggregate the per-fold distribution into a verdict-ready summary."""
    if not results:
        return {"folds": 0}
    def col(side, key):
        return np.array([getattr(f, side)[key] for f in results], dtype=float)
    raw_ir, meta_ir = col("raw", "ir"), col("meta", "ir")
    raw_mean, meta_mean = col("raw", "mean"), col("meta", "mean")
    raw_win, meta_win = col("raw", "win"), col("meta", "win")
    # meta with zero taken trades contributes 0 — count abstentions separately.
    n_take = np.array([f.n_take for f in results])
    n_test = np.array([f.n_test for f in results])
    return {
        "folds": len(results),
        "raw_ir_mean": float(raw_ir.mean()), "raw_ir_std": float(raw_ir.std(ddof=1)),
        "meta_ir_mean": float(meta_ir.mean()), "meta_ir_std": float(meta_ir.std(ddof=1)),
        "ir_uplift_mean": float((meta_ir - raw_ir).mean()),
        "ir_uplift_pos_frac": float(((meta_ir - raw_ir) > 0).mean()),
        "raw_mean_ret": float(raw_mean.mean()), "meta_mean_ret": float(meta_mean.mean()),
        "raw_win": float(raw_win.mean()), "meta_win": float(meta_win.mean()),
        "avg_kept_frac": float((n_take / np.maximum(n_test, 1)).mean()),
    }
