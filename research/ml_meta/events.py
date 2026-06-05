"""Replay the mean-reversion PRIMARY strategy PIT to emit labeling events.

Mirrors `src/strategies/mean_reversion.py::generate_mean_reversion_signal`
(long: close<BBL & StochRSI<os & RSI<os & mid>close & ADX<max & room-to-mean;
short: mirror). Indicators are computed with pandas-ta-classic, which is causal
(every value at bar t uses only bars ≤ t), so a signal at t0 is decidable at t0's
close — the entry the triple-barrier labeler then prices from.

Output is the `events` frame the labeler/feature layers consume: indexed by t0,
with `side`, `trgt` (PIT vol unit for the barriers), `t_vertical` (time stop),
and a few signal-context columns for `features.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_ta_classic as ta


@dataclass
class MeanRevParams:
    rsi_len: int = 14
    adx_len: int = 14
    atr_len: int = 14
    bb_len: int = 20
    bb_std: float = 2.0
    stoch_len: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    adx_max: float = 20.0
    min_target_atr: float = 0.5
    horizon_bars: int = 24          # vertical barrier / max hold (1 day on 1h)
    sides: tuple[str, ...] = ("long", "short")


def compute_indicators(klines: pd.DataFrame, p: MeanRevParams) -> pd.DataFrame:
    """Causal indicator panel aligned to ``klines.index``."""
    df = klines
    rsi = df.ta.rsi(length=p.rsi_len)
    atr = df.ta.atr(length=p.atr_len)                       # price units (Wilder)
    adx = df.ta.adx(length=p.adx_len)[f"ADX_{p.adx_len}"]
    bb = df.ta.bbands(length=p.bb_len, std=p.bb_std)
    stoch = df.ta.stochrsi(length=p.stoch_len)
    stoch_k = stoch[[c for c in stoch.columns if c.startswith("STOCHRSIk")][0]]
    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]
    out["rsi"] = rsi
    out["atr"] = atr
    out["adx"] = adx
    out["bbl"] = bb[f"BBL_{p.bb_len}_{p.bb_std}"]
    out["bbm"] = bb[f"BBM_{p.bb_len}_{p.bb_std}"]
    out["bbu"] = bb[f"BBU_{p.bb_len}_{p.bb_std}"]
    out["stoch_k"] = stoch_k
    return out


def mean_reversion_events(klines: pd.DataFrame, p: MeanRevParams | None = None) -> pd.DataFrame:
    """Emit one row per primary entry. Index = t0 (signal bar). Columns:
    side, trgt, t_vertical + signal-context features (stretch, rsi, adx, atr_pct,
    bb_width, dist_to_mid). All values are PIT (known at t0's close)."""
    p = p or MeanRevParams()
    ind = compute_indicators(klines, p).dropna()
    if ind.empty:
        return _empty_events()

    c, atr, mid = ind["close"], ind["atr"], ind["bbm"]
    room_ok = (mid - c).abs() >= p.min_target_atr * atr
    adx_ok = ind["adx"] <= p.adx_max

    long_mask = (
        (c < ind["bbl"]) & (ind["stoch_k"] < p.stoch_oversold)
        & (ind["rsi"] < p.rsi_oversold) & (mid > c) & adx_ok & room_ok
    ) if "long" in p.sides else pd.Series(False, index=ind.index)
    short_mask = (
        (c > ind["bbu"]) & (ind["stoch_k"] > p.stoch_overbought)
        & (ind["rsi"] > p.rsi_overbought) & (mid < c) & adx_ok & room_ok
    ) if "short" in p.sides else pd.Series(False, index=ind.index)

    side = pd.Series(index=ind.index, dtype=object)
    side[long_mask] = "long"
    side[short_mask] = "short"
    ev_idx = ind.index[long_mask | short_mask]
    if len(ev_idx) == 0:
        return _empty_events()

    # Vertical barrier = horizon_bars ahead on the SAME grid (clip at the end).
    grid = klines.index
    pos = grid.get_indexer(ev_idx)
    vpos = (pos + p.horizon_bars).clip(max=len(grid) - 1)
    t_vert = grid[vpos]
    # Drop events whose horizon is truncated to <1 bar (too close to series end).
    keep = vpos > pos

    e = ind.loc[ev_idx]
    sd = side.loc[ev_idx]
    sign = sd.map({"long": 1.0, "short": -1.0})
    stretch = pd.Series(0.0, index=ev_idx)
    stretch[sd == "long"] = ((e["bbl"] - e["close"]) / e["atr"])[sd == "long"]
    stretch[sd == "short"] = ((e["close"] - e["bbu"]) / e["atr"])[sd == "short"]

    out = pd.DataFrame({
        "side": sign.values,
        "trgt": (e["atr"] / e["close"]).values,                 # PIT vol unit
        "t_vertical": t_vert,
        "stretch": stretch.clip(lower=0.0).values,
        "rsi": e["rsi"].values,
        "adx": e["adx"].values,
        "atr_pct": (e["atr"] / e["close"]).values,
        "bb_width": ((e["bbu"] - e["bbl"]) / e["bbm"]).values,
        "dist_to_mid": ((e["bbm"] - e["close"]) / e["close"]).values,
    }, index=ev_idx)
    return out[keep]


def _empty_events() -> pd.DataFrame:
    cols = ["side", "trgt", "t_vertical", "stretch", "rsi", "adx",
            "atr_pct", "bb_width", "dist_to_mid"]
    return pd.DataFrame(columns=cols)
