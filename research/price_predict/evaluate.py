"""Standalone readout: directional skill + after-cost economics, OOS.

Two questions, both answered on the pooled walk-forward predictions:

  1. SKILL — does sign(pred) beat a coin flip? Directional accuracy with a
     binomial z vs 0.50. (We report z, not a naive t over correlated cells; even
     so, pooled coins share weeks so independence is imperfect — treat z as
     optimistic.)
  2. MONEY — trade position = sign(pred) each week, leg return =
     sign(pred)*y − cost. Mean bps/leg, t, win%, weekly Sharpe. A book is only
     interesting if MONEY is positive AFTER cost, not merely if SKILL > 0.50.

Baselines (momentum, always_long) are scored the same way: the learned models
must clear BOTH the coin flip AND the trivial momentum/up-drift books to count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST_BPS = 10.0           # round-trip cost per weekly leg (matches the funding probe)


def _dir_acc(pred: np.ndarray, y: np.ndarray) -> tuple[int, int, float, float]:
    """(n, hits, accuracy, binomial z vs 0.5). Drops y==0 (no direction)."""
    m = y != 0
    pred, y = pred[m], y[m]
    n = len(y)
    if n == 0:
        return 0, 0, 0.0, 0.0
    hits = int((np.sign(pred) == np.sign(y)).sum())
    acc = hits / n
    z = (hits - 0.5 * n) / np.sqrt(0.25 * n)
    return n, hits, acc, z


def _econ(pred: np.ndarray, y: np.ndarray, cost_bps: float) -> dict:
    """After-cost weekly long/short economics for a sign(pred) book."""
    pos = np.sign(pred)
    leg = pos * y - cost_bps / 1e4
    n = len(leg)
    if n == 0:
        return dict(n=0, bps=0.0, t=0.0, win=0.0, sharpe=0.0)
    mean = float(leg.mean())
    std = float(leg.std(ddof=1)) if n > 1 else 0.0
    return dict(
        n=n,
        bps=mean * 1e4,
        t=(mean / (std / np.sqrt(n))) if std > 0 else 0.0,
        win=float((leg > 0).mean()),
        sharpe=(mean / std * np.sqrt(52)) if std > 0 else 0.0,   # weekly→annual
    )


def summarize(preds: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Per-model skill + money table over all OOS predictions."""
    rows = []
    for name, g in preds.groupby("model"):
        p, y = g["pred"].to_numpy(), g["y"].to_numpy()
        n, hits, acc, z = _dir_acc(p, y)
        e = _econ(p, y, cost_bps)
        rows.append(dict(model=name, n=n, dir_acc=acc, z=z,
                         net_bps=e["bps"], t=e["t"], win=e["win"],
                         sharpe=e["sharpe"]))
    return pd.DataFrame(rows).set_index("model").sort_values("dir_acc", ascending=False)


def per_coin(preds: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per-coin directional accuracy for one model (concentration check)."""
    g = preds[preds["model"] == model]
    rows = []
    for coin, c in g.groupby("coin"):
        n, hits, acc, z = _dir_acc(c["pred"].to_numpy(), c["y"].to_numpy())
        rows.append(dict(coin=coin, n=n, dir_acc=acc, z=z))
    return pd.DataFrame(rows).set_index("coin").sort_values("dir_acc")


def dedrift(preds: pd.DataFrame, model: str, cost_bps: float = COST_BPS) -> dict:
    """Skill after removing each week's cross-sectional MEAN return.

    A model can score >0.50 directional just by tilting with the market's drift.
    Demeaning every week's returns across coins kills that common move, leaving
    only "can it pick which coin beats the cross-section?". If skill survives here
    it is real; if it collapses to ~0.50, the headline was drift-harvesting.
    """
    g = preds[preds["model"] == model].copy()
    if g.empty:
        return {}
    g["yd"] = g.groupby("t")["y"].transform(lambda s: s - s.mean())
    p, yd = g["pred"].to_numpy(), g["yd"].to_numpy()
    n, hits, acc, z = _dir_acc(p, yd)
    e = _econ(p, yd, cost_bps)
    return dict(n=n, dir_acc=round(acc, 3), z=round(z, 2),
                net_bps=round(e["bps"], 1), t=round(e["t"], 2))


def temporal_halves(preds: pd.DataFrame, model: str,
                    cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Split the OOS span in two — does skill/money persist or flip across time?"""
    g = preds[preds["model"] == model].sort_values("t")
    if g.empty:
        return pd.DataFrame()
    mid = g["t"].iloc[len(g) // 2]
    rows = []
    for label, d in (("half1", g[g["t"] < mid]), ("half2", g[g["t"] >= mid]),
                     ("full", g)):
        p, y = d["pred"].to_numpy(), d["y"].to_numpy()
        n, hits, acc, z = _dir_acc(p, y)
        e = _econ(p, y, cost_bps)
        rows.append(dict(span=label, n=n, dir_acc=acc, z=z,
                         net_bps=e["bps"], t=e["t"]))
    return pd.DataFrame(rows).set_index("span")
