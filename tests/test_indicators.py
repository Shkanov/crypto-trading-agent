from __future__ import annotations

import random

from src.models.types import Kline
from src.tools.indicators import IndicatorEngine


def _kline(symbol: str, tf: str, c: float, h: float | None = None, l: float | None = None,
           v: float = 1.0, taker_buy: float = 0.5) -> Kline:
    return Kline(
        symbol=symbol, timeframe=tf, open_time=0, close_time=0,
        open=c, high=h or c, low=l or c, close=c,
        volume=v, quote_volume=c * v, trades=10, taker_buy_volume=taker_buy,
        is_closed=True,
    )


def test_engine_handles_warmup_without_errors():
    e = IndicatorEngine()
    random.seed(0)
    price = 100.0
    for _ in range(120):
        price *= 1 + random.uniform(-0.005, 0.005)
        k = _kline("BTCUSDT", "5m", price, h=price * 1.001, l=price * 0.999, v=10, taker_buy=6)
        e.get("BTCUSDT", "5m").on_closed_kline(k)
    snap = e.latest("BTCUSDT", "5m")
    assert snap is not None
    assert snap.ema21 is not None and snap.ema55 is not None
    assert snap.rsi14 is not None
    assert snap.atr14 is not None and snap.atr14 > 0
    assert snap.bb_upper is not None and snap.bb_lower is not None
    assert snap.vwap is not None


def test_ema_monotonic_for_trending_series():
    e = IndicatorEngine()
    for i in range(100):
        c = 100.0 + i
        e.get("BTCUSDT", "5m").on_closed_kline(_kline("BTCUSDT", "5m", c, c + 0.5, c - 0.5))
    snap = e.latest("BTCUSDT", "5m")
    assert snap is not None and snap.ema21 is not None and snap.ema55 is not None
    assert snap.ema21 > snap.ema55  # uptrend → fast EMA above slow EMA


def test_rsi_extreme_for_pure_uptrend():
    e = IndicatorEngine()
    for i in range(50):
        c = 100.0 + i
        e.get("BTCUSDT", "5m").on_closed_kline(_kline("BTCUSDT", "5m", c))
    snap = e.latest("BTCUSDT", "5m")
    assert snap is not None and snap.rsi14 is not None
    assert snap.rsi14 > 70  # uptrend should push RSI high
