"""Overlay readout: predictor as a Δfunding side-filter.

The funding book picks a leg (coin, side ±1) purely from the funding-rate change.
This asks: if we ALSO require the price predictor to agree with the leg's
direction — keep the leg only when sign(pred_next_week_return) == side — do the
SURVIVING legs pay better than the raw book, OOS?

Mechanics:
  * dfunding_events replays the weekly rebalances on the funding universe
    (identical to the live decision functions),
  * each leg's economics = price log-return over the one-week hold + funding
    return − cost (same accounting as run_dfunding),
  * the predictor's OOS forecast for that coin/week is matched by taking the
    first weekly block ending at/after the rebalance instant (the forecast made
    from data up to t0, realised over the coming week — aligned to the hold),
  * compare raw vs sign-agreement-filtered per-leg net economics on the SAME
    OOS legs. Coverage (how many legs even HAVE a matching OOS prediction) is
    reported — thin-history funding alts often won't, and that is an honest limit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ml_meta.funding_events import DFundingParams, dfunding_events
from research.ml_meta.labeling import triple_barrier_labels
from research.ml_meta.run_dfunding import _funding_return

COST_BPS = 10.0


def funding_legs(price: dict[str, pd.DataFrame],
                 funding: dict[str, list[tuple[int, float]]],
                 p: DFundingParams) -> pd.DataFrame:
    """Replay Δfunding → one row per leg: t0, sym, side, price, fund (returns)."""
    ev = dfunding_events(funding, price, p)
    rows = []
    for sym, e in ev.items():
        lab = triple_barrier_labels(price[sym]["close"], e, pt=0, sl=0, cost_bps=0)
        for t0 in e.index:
            t1 = lab.t1.loc[t0]
            side = float(e.loc[t0, "side"])
            t0ms = int(pd.Timestamp(t0).value // 1_000_000)
            t1ms = int(pd.Timestamp(t1).value // 1_000_000)
            rows.append(dict(t0=pd.Timestamp(t0), sym=sym, side=side,
                             price=float(lab.ret.loc[t0]),
                             fund=_funding_return(funding[sym], t0ms, t1ms, side)))
    return pd.DataFrame(rows)


def match_predictions(legs: pd.DataFrame, preds: pd.DataFrame, model: str,
                      step_days: int = 7) -> pd.DataFrame:
    """Attach each leg the predictor's forecast for its coin/week.

    For a leg entered at t0, take the prediction for that coin whose target week
    ends in [t0, t0 + step_days] — the forecast realised over the coming hold.
    Legs with no matching OOS prediction get pred=NaN (excluded from the filter).
    """
    pm = preds[preds["model"] == model]
    tol = pd.Timedelta(days=step_days)
    out = legs.copy()
    pred_vals = np.full(len(out), np.nan)
    for i, row in enumerate(out.itertuples(index=False)):
        cand = pm[(pm["coin"] == row.sym) & (pm["t"] >= row.t0)
                  & (pm["t"] <= row.t0 + tol)]
        if not cand.empty:
            pred_vals[i] = cand.sort_values("t")["pred"].iloc[0]
    out["pred"] = pred_vals
    return out


def _net(df: pd.DataFrame, cost_bps: float) -> dict:
    if df.empty:
        return dict(n=0, bps=0.0, t=0.0, win=0.0)
    net = (df["price"] + df["fund"] - cost_bps / 1e4).to_numpy()
    n = len(net)
    std = net.std(ddof=1) if n > 1 else 0.0
    return dict(n=n, bps=float(net.mean() * 1e4),
                t=float(net.mean() / (std / np.sqrt(n))) if std > 0 else 0.0,
                win=float((net > 0).mean()))


def overlay_compare(legs: pd.DataFrame, model: str,
                    cost_bps: float = COST_BPS) -> dict:
    """Raw vs sign-agreement-filtered economics, on legs that HAVE a prediction.

    Keep a leg iff sign(pred) == side (predictor agrees with the funding leg's
    direction). Reports coverage + both books' per-leg net economics.
    """
    covered = legs[legs["pred"].notna()].copy()
    if covered.empty:
        return dict(model=model, coverage=0, total_legs=len(legs))
    agree = np.sign(covered["pred"].to_numpy()) == covered["side"].to_numpy()
    return dict(
        model=model,
        total_legs=len(legs),
        coverage=len(covered),
        kept=int(agree.sum()),
        raw=_net(covered, cost_bps),
        filtered=_net(covered[agree], cost_bps),
    )
