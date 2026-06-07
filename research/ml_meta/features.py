"""PIT feature matrix at each primary event.

Three families, all causal (every value at t0 uses bars ≤ t0):
  - signal-context : carried from events.py (stretch, rsi, adx, atr_pct, ...)
  - symbol-context : the instrument's own recent return / realized vol / trend
  - market regime  : BTC vol & trend + cross-sectional dispersion of the panel

The non-linearity thesis lives in the interaction of these (e.g. mean-rev pays
only when ADX is low AND BTC vol is low AND dispersion is high). Trees find that;
a linear allocator can't. `is_long` is included so the model can learn
side-conditional behavior.
"""
from __future__ import annotations

import pandas as pd


def _causal_symbol_features(k: pd.DataFrame) -> pd.DataFrame:
    ret = k["close"].pct_change()
    f = pd.DataFrame(index=k.index)
    f["ret24"] = k["close"].pct_change(24)
    f["vol24"] = ret.rolling(24).std()
    f["vol72"] = ret.rolling(72).std()
    f["trend50"] = k["close"] / k["close"].rolling(50).mean() - 1.0
    return f


def market_features(panel: dict[str, pd.DataFrame], btc_sym: str = "BTCUSDT") -> pd.DataFrame:
    """Causal market-regime features on the union grid of the panel."""
    grid = None
    for k in panel.values():
        grid = k.index if grid is None else grid.union(k.index)
    m = pd.DataFrame(index=grid)
    if btc_sym in panel:
        bf = _causal_symbol_features(panel[btc_sym]).reindex(grid).ffill()
        m["btc_ret24"] = bf["ret24"]
        m["btc_vol24"] = bf["vol24"]
        m["btc_trend50"] = bf["trend50"]
    rets24 = pd.DataFrame(
        {s: k["close"].pct_change(24) for s, k in panel.items()}
    ).reindex(grid)
    m["xs_dispersion"] = rets24.std(axis=1)
    m["xs_mean_ret"] = rets24.mean(axis=1)
    return m


SIGNAL_COLS = ["stretch", "rsi", "adx", "atr_pct", "bb_width", "dist_to_mid"]

# Common (primary-agnostic) feature families appended after the signal columns.
SYMBOL_COLS = ["ret24", "vol24", "vol72", "trend50"]
MARKET_COLS = ["btc_ret24", "btc_vol24", "btc_trend50", "xs_dispersion", "xs_mean_ret"]
TIME_COLS = ["hour", "dow"]


def feature_cols(signal_cols: list[str] = SIGNAL_COLS,
                 *, include_time: bool = True) -> list[str]:
    """The model's feature column list for a given primary's signal columns.
    `is_long` lets the model learn side-conditional behavior. Weekly primaries
    (e.g. Δfunding) rebalance on a fixed clock, so hour/dow are near-constant
    and add nothing — drop them with include_time=False."""
    cols = list(signal_cols) + ["is_long"] + SYMBOL_COLS + MARKET_COLS
    if include_time:
        cols += TIME_COLS
    return cols


# Default (mean-rev) feature list — kept for the Phase-1 driver and evaluate.py.
FEATURE_COLS = feature_cols(SIGNAL_COLS)


def build_features(events_by_symbol: dict[str, pd.DataFrame],
                   panel: dict[str, pd.DataFrame],
                   btc_sym: str = "BTCUSDT",
                   *, signal_cols: list[str] = SIGNAL_COLS) -> pd.DataFrame:
    """One row per event with the feature columns plus carry columns
    (sym, t0, side, trgt, t_vertical). Index is a clean RangeIndex. Generic
    over `signal_cols` so any primary's events frame (mean-rev, Δfunding, ...)
    can carry its own signal-context columns through unchanged."""
    market = market_features(panel, btc_sym)
    frames = []
    for sym, ev in events_by_symbol.items():
        if ev is None or len(ev) == 0:
            continue
        sf = _causal_symbol_features(panel[sym]).reindex(ev.index)
        mf = market.reindex(ev.index)
        df = pd.DataFrame(index=ev.index)
        df["sym"] = sym
        df["t0"] = ev.index
        df["side"] = ev["side"].values
        df["trgt"] = ev["trgt"].values
        df["t_vertical"] = ev["t_vertical"].values
        for c in signal_cols:
            df[c] = ev[c].values
        df["is_long"] = (ev["side"].values > 0).astype(float)
        for c in SYMBOL_COLS:
            df[c] = sf[c].values
        for c in MARKET_COLS:
            df[c] = mf[c].values
        df["hour"] = ev.index.hour.astype(float)
        df["dow"] = ev.index.dayofweek.astype(float)
        frames.append(df)
    if not frames:
        carry = ["sym", "t0", "side", "trgt", "t_vertical"]
        return pd.DataFrame(columns=carry + feature_cols(signal_cols))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("t0").reset_index(drop=True)
