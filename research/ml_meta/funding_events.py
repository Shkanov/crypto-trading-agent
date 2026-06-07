"""Replay the Δfunding cross-sectional PRIMARY to emit labeling events.

This is the Phase-2 primary: the one strategy with a *validated raw positive
edge* (CPCV PASS on top-30/50). Meta-labeling a profitable base asks a sharper
question than Phase-1's mean-rev did — "which of the legs this rule already
picks are the ones that actually pay, and can a model skip the rest?"

Mechanics mirror `src/strategies/dfunding_carry.py` EXACTLY by reusing the live
decision functions (`funding_window_change`, `rank_for_carry`) — live/backtest
parity by construction:

  Every `rebalance_hours` (default 168h = 1 week), on a fixed clock:
    1. For each symbol compute Δfunding = mean(funding, recent window) −
       mean(funding, prior window).  PIT: only funding events strictly < t.
    2. Rank the cross-section.  LONG the top-N highest Δfunding (funding
       accelerating up), SHORT the bottom-N lowest.
    3. Each selected leg becomes ONE event at t0 (the price bar at/just before
       the rebalance instant), side ±1, held to the next rebalance (vertical
       barrier at t0 + rebalance_hours).

Unlike the per-symbol mean-rev events, selection here is *cross-sectional*: a
leg exists only because it ranked at an extreme of the panel that week. The
features carry that context (Δfunding value, funding level, cross-sectional
rank + dispersion) so the meta-model can learn which extremes are real.

Universe note: Phase-2 uses the FIXED 16-symbol cache panel, not the live
PIT-by-volume top-30. That trades some breadth + survivorship realism for a
clean reproducible study; the meta-vs-raw *relative* comparison (the actual
question) is unaffected. Flagged again in run_dfunding.py.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.strategies.funding_carry import (
    CarryParams,
    funding_window_change,
    rank_for_carry,
)

EIGHT_H_MS = 8 * 3_600_000

# Live dfunding_carry EXCLUDES the majors from the candidate set
# (dfunding_carry.py:157). The edge is documented to live in lower-cap higher-OI
# alts; including majors lets the PIT volume rank fill the book with BTC/ETH/etc.
# — the exact segment the live strategy drops. Mirror live here for parity.
MAJORS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"})

SIGNAL_COLS = ["dfunding", "funding_now", "funding_abs",
               "dfunding_rank_pct", "xs_funding_disp", "rv_week"]


@dataclass(frozen=True)
class DFundingParams:
    window_cycles: int = 21          # 21×8h = 168h, matches live default
    top_n: int = 3                   # legs per side
    rebalance_hours: int = 168       # weekly
    rv_lookback_bars: int = 168      # trailing window for realized-vol barrier unit
    universe_size: int = 30          # PIT top-N by 24h volume (matches live default)
    vol_lookback_bars: int = 24      # trailing window for the volume universe

    @property
    def window_hours(self) -> int:
        return self.window_cycles * 8

    def carry_params(self) -> CarryParams:
        return CarryParams(top_n=self.top_n)


def _window_means(events: list[tuple[int, float]], ts_ms: int,
                  window_hours: int) -> Optional[tuple[float, float]]:
    """(recent_mean, prior_mean) of funding over the two trailing windows
    ending at ts_ms. PIT: only events strictly before ts_ms. None if either
    window is empty — mirrors funding_window_change's eligibility."""
    w_ms = window_hours * 3_600_000
    recent = [r for (t, r) in events if ts_ms - w_ms <= t < ts_ms]
    prior = [r for (t, r) in events if ts_ms - 2 * w_ms <= t < ts_ms - w_ms]
    if not recent or not prior:
        return None
    return sum(recent) / len(recent), sum(prior) / len(prior)


def _realized_vol_week(close: pd.Series, lookback: int) -> pd.Series:
    """Trailing realized vol scaled to a one-week horizon (the barrier unit).
    Causal: rolling std of hourly returns over `lookback` bars × √lookback."""
    ret = close.pct_change()
    return ret.rolling(lookback).std() * np.sqrt(lookback)


def dfunding_events(funding_panel: dict[str, list[tuple[int, float]]],
                    price_panel: dict[str, pd.DataFrame],
                    p: DFundingParams | None = None
                    ) -> dict[str, pd.DataFrame]:
    """Replay weekly Δfunding rebalances → {symbol: events frame}.

    Each frame is indexed by t0 (on the price grid) with columns: side, trgt,
    t_vertical + SIGNAL_COLS. Empty symbols are omitted. The output plugs
    straight into labeling/triple_barrier and features.build_features.
    """
    p = p or DFundingParams()
    symbols = [s for s in funding_panel if s in price_panel and s not in MAJORS]
    if len(symbols) < 2 * p.top_n:
        return {}

    # Shared price grid (the 16 cached perps are aligned, but union is safe).
    grid: Optional[pd.DatetimeIndex] = None
    for s in symbols:
        idx = price_panel[s].index
        grid = idx if grid is None else grid.union(idx)
    assert grid is not None
    # Parquet may hand the index back at ms (not ns) resolution, so asi8's unit
    # is ambiguous — pin it to ns, then ns→ms, to get a true epoch-ms array.
    grid_ms = grid.as_unit("ns").asi8 // 1_000_000
    g0, g1 = int(grid_ms[0]), int(grid_ms[-1])

    rv = {s: _realized_vol_week(price_panel[s]["close"], p.rv_lookback_bars)
          for s in symbols}
    # Quote-volume aligned to the grid, for the PIT volume universe.
    qv = {s: price_panel[s]["quote_volume"].reindex(grid).fillna(0.0).to_numpy()
          for s in symbols}

    rb_ms = p.rebalance_hours * 3_600_000
    # First rebalance once two funding windows exist; last so the full hold fits.
    first = g0 + 2 * p.window_hours * 3_600_000
    rows: dict[str, list[dict]] = defaultdict(list)

    t = first
    while t + rb_ms <= g1:
        pos0 = int(np.searchsorted(grid_ms, t, side="right") - 1)
        if pos0 < p.vol_lookback_bars:
            t += rb_ms
            continue
        # PIT volume universe: rank candidates by trailing 24h quote volume AS OF
        # this rebalance, keep the top `universe_size`. Replicates the live
        # by-volume universe and removes the "today's leaders applied
        # retroactively" selection look-ahead. Residual survivorship: the
        # candidate pool itself is names that exist across the full window.
        lo = pos0 - p.vol_lookback_bars + 1
        vol_now = {s: float(qv[s][lo:pos0 + 1].sum()) for s in symbols}
        vol_now = {s: v for s, v in vol_now.items() if v > 0}
        pool = sorted(vol_now, key=lambda s: vol_now[s],
                      reverse=True)[: p.universe_size]
        pool = set(pool)

        # PIT Δfunding signal per symbol in the volume pool (past funding only).
        sig: dict[str, float] = {}
        means: dict[str, tuple[float, float]] = {}
        for s in pool:
            df = funding_window_change(funding_panel[s], t, p.window_hours)
            mw = _window_means(funding_panel[s], t, p.window_hours)
            if df is not None and mw is not None:
                sig[s] = df
                means[s] = mw
        if len(sig) >= 2 * p.top_n:
            longs, shorts = rank_for_carry(sig, p.carry_params())
            if longs and shorts:
                vals = np.array(list(sig.values()))
                disp = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                order = sorted(sig, key=lambda k: sig[k])
                rank_pct = {s: i / (len(order) - 1) for i, s in enumerate(order)}

                # Snap the hold's end to the grid (pos0 already snapped above).
                posv = int(np.searchsorted(grid_ms, t + rb_ms, side="right") - 1)
                if 0 <= pos0 < posv <= len(grid) - 1:
                    t0, t_vert = grid[pos0], grid[posv]
                    for sym, side in [(s, 1.0) for s in longs] + \
                                     [(s, -1.0) for s in shorts]:
                        if t0 not in price_panel[sym].index:
                            continue
                        recent_mean, _prior = means[sym]
                        rvv = rv[sym].get(t0, np.nan)
                        rows[sym].append({
                            "t0": t0,
                            "side": side,
                            "trgt": float(rvv) if rvv == rvv and rvv > 0 else 0.02,
                            "t_vertical": t_vert,
                            "dfunding": float(sig[sym]),
                            "funding_now": float(recent_mean),
                            "funding_abs": float(abs(recent_mean)),
                            "dfunding_rank_pct": float(rank_pct[sym]),
                            "xs_funding_disp": disp,
                            "rv_week": float(rvv) if rvv == rvv else 0.0,
                        })
        t += rb_ms

    out: dict[str, pd.DataFrame] = {}
    for sym, rs in rows.items():
        if not rs:
            continue
        f = pd.DataFrame(rs).set_index("t0").sort_index()
        out[sym] = f
    return out
