"""Choppiness Index tests + regime-gate behavior."""
from __future__ import annotations

import math

from src.models.types import Kline
from src.tools.indicators import IndicatorEngine


def _bar(t: int, o: float, h: float, l: float, c: float) -> Kline:
    return Kline(
        symbol="X", timeframe="15m",
        open_time=t, close_time=t + 900_000,
        open=o, high=h, low=l, close=c,
        volume=1.0, quote_volume=1.0, trades=1,
        taker_buy_volume=0.5, is_closed=True,
    )


def test_choppiness_low_on_pure_trend():
    """Pure trending sequence (no retracement, narrow bars) → CHOP < 38.2."""
    eng = IndicatorEngine()
    snap = None
    px = 100.0
    for i in range(40):
        # Each bar opens at prior close, gains 1%, no overlap with prior bar.
        nxt = px * 1.01
        snap = eng.get("X", "15m").on_closed_kline(_bar(i * 900_000, px, nxt, px, nxt))
        px = nxt
    assert snap is not None
    assert snap.choppiness14 is not None
    assert snap.choppiness14 < 38.2, f"trend should give low CHOP, got {snap.choppiness14}"


def test_choppiness_high_on_choppy_market():
    """Wide overlapping bars in a sideways range → CHOP > 50."""
    eng = IndicatorEngine()
    snap = None
    base = 100.0
    for i in range(40):
        # Every bar swings ±2% but closes back near base — pure chop
        h = base + 2.0
        l = base - 2.0
        c = base + (0.5 if i % 2 == 0 else -0.5)
        snap = eng.get("X", "15m").on_closed_kline(_bar(i * 900_000, base, h, l, c))
    assert snap is not None
    assert snap.choppiness14 is not None
    assert snap.choppiness14 > 50.0, f"chop should give high CHOP, got {snap.choppiness14}"
