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
FEATURE_COLS = (
    SIGNAL_COLS
    + ["is_long", "ret24", "vol24", "vol72", "trend50"]
    + ["btc_ret24", "btc_vol24", "btc_trend50", "xs_dispersion", "xs_mean_ret"]
    + ["hour", "dow"]
)


def build_features(events_by_symbol: dict[str, pd.DataFrame],
                   panel: dict[str, pd.DataFrame],
                   btc_sym: str = "BTCUSDT") -> pd.DataFrame:
    """One row per event with FEATURE_COLS plus carry columns
    (sym, t0, side, trgt, t_vertical). Index is a clean RangeIndex."""
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
        for c in SIGNAL_COLS:
            df[c] = ev[c].values
        df["is_long"] = (ev["side"].values > 0).astype(float)
        for c in ["ret24", "vol24", "vol72", "trend50"]:
            df[c] = sf[c].values
        for c in ["btc_ret24", "btc_vol24", "btc_trend50", "xs_dispersion", "xs_mean_ret"]:
            df[c] = mf[c].values
        df["hour"] = ev.index.hour.astype(float)
        df["dow"] = ev.index.dayofweek.astype(float)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["sym", "t0", "side", "trgt", "t_vertical"] + FEATURE_COLS)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("t0").reset_index(drop=True)
