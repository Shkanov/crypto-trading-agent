"""Triple-barrier meta-labels + average-uniqueness sample weights.

López de Prado, *Advances in Financial Machine Learning* (AFML), ch. 3 (labeling)
and ch. 4 (sample weights). These are pure functions over a single symbol's close
series — no I/O, no global state — so they are deterministic and unit-testable.

Leakage discipline baked in here:
  - Entry price is `close[t0]` (the price known AT the decision bar). The barrier
    scan reads only bars STRICTLY AFTER t0, so a label never peeks at its own
    decision bar's future within the same step.
  - The label window is `[t0, t1]` where t1 is the first barrier touch (profit,
    stop, or vertical/time barrier). `t1` is exported so the CV layer can PURGE
    any training event whose window overlaps a test fold.
  - "Win" (bin=1) requires the directional return to clear the round-trip COST,
    not merely be positive. A meta-model trained on cost-naive labels would learn
    to take trades that lose money after fees.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TripleBarrierResult:
    t1: pd.Series      # first-touch timestamp per event (indexed by t0)
    ret: pd.Series     # realized DIRECTIONAL gross return at t1 (side-adjusted)
    bin: pd.Series     # meta-label: 1 if (ret - cost) > min_ret else 0
    touch: pd.Series   # which barrier touched first: 'pt' | 'sl' | 'vertical'


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DataFrame,
    pt: float,
    sl: float,
    *,
    cost_bps: float = 0.0,
    min_ret: float = 0.0,
) -> TripleBarrierResult:
    """Label each primary event by the first of three barriers it touches.

    Parameters
    ----------
    close : pd.Series
        Close prices indexed by a monotonic DatetimeIndex (one symbol, one TF).
    events : pd.DataFrame
        Indexed by ``t0`` (decision timestamps, must be a subset of ``close.index``).
        Required columns:
          - ``side`` : +1 long / -1 short (the PRIMARY model's bet direction)
          - ``trgt`` : target return unit at t0 (e.g. forward vol estimate). The
                       profit barrier sits at ``pt*trgt`` and the stop at
                       ``-sl*trgt`` in directional-return space.
          - ``t_vertical`` : vertical-barrier timestamp (time stop). Must be > t0.
    pt, sl : float
        Profit-take and stop-loss multiples of ``trgt`` (>= 0; 0 disables that
        horizontal barrier).
    cost_bps : float
        Round-trip cost in bps, subtracted from the realized return before the
        win/lose decision. So bin=1 means "profitable AFTER costs".
    min_ret : float
        Extra return cushion required to call a win (default 0).

    Returns
    -------
    TripleBarrierResult with everything indexed by t0.
    """
    if not close.index.is_monotonic_increasing:
        raise ValueError("close index must be sorted ascending")
    for col in ("side", "trgt", "t_vertical"):
        if col not in events.columns:
            raise ValueError(f"events missing required column {col!r}")
    if (events["t_vertical"] <= events.index).any():
        raise ValueError("every t_vertical must be strictly after its t0 (no zero/negative horizon)")

    cost = cost_bps / 1e4
    idx = close.index
    t1_out, ret_out, touch_out = [], [], []

    for t0, ev in events.iterrows():
        side = float(ev["side"])
        trgt = float(ev["trgt"])
        t_vert = ev["t_vertical"]
        entry = float(close.loc[t0])
        # Path STRICTLY AFTER the decision bar, up to and including the time stop.
        seg = close.loc[(idx > t0) & (idx <= t_vert)]
        if seg.empty:
            # Degenerate: no bars between t0 and the time stop. Skip.
            t1_out.append(t_vert); ret_out.append(0.0); touch_out.append("vertical")
            continue
        # Directional cumulative return along the path.
        dr = side * (seg.values / entry - 1.0)
        up = pt * trgt if pt > 0 else np.inf
        dn = -sl * trgt if sl > 0 else -np.inf

        hit_pt = np.where(dr >= up)[0]
        hit_sl = np.where(dr <= dn)[0]
        i_pt = hit_pt[0] if hit_pt.size else None
        i_sl = hit_sl[0] if hit_sl.size else None

        candidates = []
        if i_pt is not None:
            candidates.append((i_pt, "pt"))
        if i_sl is not None:
            candidates.append((i_sl, "sl"))
        if candidates:
            i_hit, which = min(candidates, key=lambda x: x[0])
            t1 = seg.index[i_hit]
            r = float(dr[i_hit])
            touch = which
        else:
            t1 = seg.index[-1]
            r = float(dr[-1])
            touch = "vertical"

        t1_out.append(t1); ret_out.append(r); touch_out.append(touch)

    t1_s = pd.Series(t1_out, index=events.index, name="t1")
    ret_s = pd.Series(ret_out, index=events.index, name="ret")
    bin_s = ((ret_s - cost) > min_ret).astype(int).rename("bin")
    touch_s = pd.Series(touch_out, index=events.index, name="touch")
    return TripleBarrierResult(t1=t1_s, ret=ret_s, bin=bin_s, touch=touch_s)


def average_uniqueness(bar_index: pd.DatetimeIndex,
                       t0: pd.Series,
                       t1: pd.Series) -> pd.Series:
    """Average uniqueness of each label as a SAMPLE WEIGHT (AFML 4.2).

    Forward-horizon labels overlap in time, so consecutive samples are NOT IID —
    treating them as IID inflates effective sample size and overstates confidence.
    For each bar, ``concurrency`` = number of labels whose window spans it. A
    label's average uniqueness = mean of ``1/concurrency`` over its own window;
    overlapping labels get downweighted. Returned indexed by t0.

    ``t0`` and ``t1`` must be timestamps lying on ``bar_index``.
    """
    if len(t0) != len(t1):
        raise ValueError("t0 and t1 must be the same length")
    pos = pd.Series(np.arange(len(bar_index)), index=bar_index)
    starts = pos.reindex(pd.Index(t0.values)).values
    ends = pos.reindex(pd.Index(t1.values)).values
    if np.isnan(starts).any() or np.isnan(ends).any():
        raise ValueError("every t0/t1 must lie on bar_index")
    starts = starts.astype(int); ends = ends.astype(int)

    # concurrency over the grid via a +1/-1 difference array
    conc = np.zeros(len(bar_index) + 1)
    for s, e in zip(starts, ends):
        conc[s] += 1
        conc[e + 1] -= 1
    conc = np.cumsum(conc)[:-1]
    inv = np.divide(1.0, conc, out=np.zeros(len(conc)), where=conc > 0)

    uniq = np.empty(len(t0))
    for k, (s, e) in enumerate(zip(starts, ends)):
        uniq[k] = inv[s:e + 1].mean()
    return pd.Series(uniq, index=t0.index, name="w_uniqueness")
