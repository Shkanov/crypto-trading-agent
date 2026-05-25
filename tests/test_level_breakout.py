"""Tests for the level-breakout strategy.

We bypass the orchestrator wiring and feed klines + snapshots directly into
`LevelBreakoutStrategy.on_bar`, with a stub `StrategyContext` that records
proposals into a list. This isolates the entry logic from the rest of the
system (risk gate, executor, storage) so we can pin individual rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from src.config.settings import Settings
from src.models.types import (
    FeatureVector,
    IndicatorSnapshot,
    Kline,
    PairProposal,
    Signal,
    Trade,
)
from src.services.event_bus import EventBus
from src.services.storage import Storage
from src.strategies.level_breakout import (
    LevelBreakoutParams,
    LevelBreakoutStrategy,
    _PivotBuffer,
)
from src.tools.binance_client import BinanceClient
from src.tools.indicators import IndicatorEngine


# ─────────────────────────────── helpers ───────────────────────────────


@dataclass
class _CtxStub:
    """Minimal fake context. We don't run a real EventBus / Storage etc. —
    we just record proposals and serve indicator snapshots."""
    settings: Settings
    indicators: IndicatorEngine
    proposals: list[tuple[Signal, str, int, str]] = field(default_factory=list)
    storage: Optional[Storage] = None
    bus: Optional[EventBus] = None
    binance: Optional[BinanceClient] = None
    last_price: dict[str, float] = field(default_factory=dict)
    pair_executor: Optional[object] = None

    async def propose(self, signal: Signal, market: str, leverage: int,
                       strategy_name: str) -> None:
        self.proposals.append((signal, market, leverage, strategy_name))

    async def propose_pair(self, pair: PairProposal) -> None:
        pass

    async def close_trade(self, trade_id: str, reason: str) -> None:
        pass

    def open_trades(self, strategy_name: Optional[str] = None) -> list[Trade]:
        return []

    def equity_available_usd(self, strategy_name: Optional[str] = None) -> float:
        return self.settings.account_equity_usd


def _kline(symbol: str, tf: str, ts_ms: int, open_: float, high: float,
           low: float, close: float, volume: float = 1_000.0) -> Kline:
    return Kline(
        symbol=symbol, timeframe=tf,
        open_time=ts_ms - 60_000, close_time=ts_ms,
        open=open_, high=high, low=low, close=close,
        volume=volume, quote_volume=volume * close, trades=10,
        taker_buy_volume=volume / 2, is_closed=True,
    )


def _snap(symbol: str, tf: str, close: float, *, atr: float, vol_z: float,
          rsi: float, ema55: float | None = None,
          supertrend_dir: int | None = None) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        symbol=symbol, timeframe=tf, close=close,
        atr14=atr, volume_z=vol_z, rsi14=rsi,
        ema55=ema55, supertrend_dir=supertrend_dir,
    )


def _settings() -> Settings:
    # Construct without reading env; we only use account_equity, market_type,
    # max_leverage, timeframe_list inside the strategy. Defaults are fine.
    return Settings()


def _make_strategy(**overrides) -> tuple[LevelBreakoutStrategy, _CtxStub]:
    s = _settings()
    ind = IndicatorEngine()
    ctx = _CtxStub(settings=s, indicators=ind)
    params = LevelBreakoutParams(**overrides) if overrides else LevelBreakoutParams()
    strat = LevelBreakoutStrategy(params=params, symbols=["BTCUSDT"])
    return strat, ctx


# ─────────────────────────────── pivot buffer ──────────────────────────


def test_pivot_buffer_detects_central_high():
    pb = _PivotBuffer(window=2)
    # Push 5 bars where the middle one is a clear high.
    heights = [(10, 5), (11, 6), (15, 7), (12, 6), (10, 5)]
    for h, l in heights:
        pb.push(h, l)
    assert len(pb.highs) == 1
    assert pb.highs[0].price == 15
    assert pb.highs[0].bar_ix == 2


def test_pivot_buffer_detects_central_low():
    pb = _PivotBuffer(window=2)
    # Push 5 bars where the middle one is a clear low.
    bars = [(20, 10), (19, 9), (18, 5), (19, 9), (20, 10)]
    for h, l in bars:
        pb.push(h, l)
    assert len(pb.lows) == 1
    assert pb.lows[0].price == 5
    assert pb.lows[0].bar_ix == 2


def test_pivot_buffer_lags_by_window():
    """A pivot at bar N is only recorded AFTER `window` more bars push in.
    This is the core anti-look-ahead invariant."""
    pb = _PivotBuffer(window=2)
    pb.push(10, 5)
    pb.push(11, 6)
    pb.push(15, 7)  # the eventual pivot — but we don't know yet
    assert len(pb.highs) == 0  # only 3 bars pushed, need 2*2+1=5
    pb.push(12, 6)
    assert len(pb.highs) == 0
    pb.push(10, 5)
    assert len(pb.highs) == 1  # now confirmed


# ─────────────────────────────── HTF break path ────────────────────────


@pytest.mark.asyncio
async def test_htf_break_long_fires():
    strat, ctx = _make_strategy(trendline_enabled=False)
    await strat.start(ctx)

    # Inject the HTF level by feeding a closed HTF bar.
    htf_k = _kline("BTCUSDT", "1d", 1_000_000, 100, 110, 95, 105)
    # Engine needs a snapshot; for HTF path the strategy reads k.high directly.
    await strat.on_bar(htf_k, _snap("BTCUSDT", "1d", 105, atr=2, vol_z=0, rsi=50))
    # Plant an HTF snapshot in the engine so the regime filter passes.
    ctx.indicators.get("BTCUSDT", "1d").last_snapshot = IndicatorSnapshot(  # type: ignore[assignment]
        symbol="BTCUSDT", timeframe="1d", close=120, ema55=100, supertrend_dir=1,
    )

    # Two trigger bars: first establishes "prior close ≤ level", second breaks.
    tf = "5m"
    k1 = _kline("BTCUSDT", tf, 1_010_000, 108, 109, 107, 108)
    await strat.on_bar(k1, _snap("BTCUSDT", tf, 108, atr=0.5, vol_z=1.2, rsi=55))
    assert ctx.proposals == []  # no break yet (108 < 110)

    k2 = _kline("BTCUSDT", tf, 1_010_300, 109, 112, 109, 111)
    await strat.on_bar(k2, _snap("BTCUSDT", tf, 111, atr=0.5, vol_z=1.5, rsi=58))
    assert len(ctx.proposals) == 1
    sig, market, leverage, name = ctx.proposals[0]
    assert sig.side == "long"
    assert sig.entry == 111
    assert sig.stop < 111  # below entry for long
    assert sig.take_profit > 111
    assert name == "levelbreak"


@pytest.mark.asyncio
async def test_htf_break_short_fires():
    strat, ctx = _make_strategy(trendline_enabled=False)
    await strat.start(ctx)
    htf_k = _kline("BTCUSDT", "1d", 1_000_000, 105, 110, 95, 100)
    await strat.on_bar(htf_k, _snap("BTCUSDT", "1d", 100, atr=2, vol_z=0, rsi=50))
    ctx.indicators.get("BTCUSDT", "1d").last_snapshot = IndicatorSnapshot(  # type: ignore[assignment]
        symbol="BTCUSDT", timeframe="1d", close=90, ema55=100, supertrend_dir=-1,
    )

    tf = "5m"
    k1 = _kline("BTCUSDT", tf, 1_010_000, 97, 98, 95.5, 96)
    await strat.on_bar(k1, _snap("BTCUSDT", tf, 96, atr=0.5, vol_z=1.2, rsi=45))
    assert ctx.proposals == []  # 96 > 95, no break

    k2 = _kline("BTCUSDT", tf, 1_010_300, 95.5, 96, 93, 93.5)
    await strat.on_bar(k2, _snap("BTCUSDT", tf, 93.5, atr=0.5, vol_z=1.5, rsi=40))
    assert len(ctx.proposals) == 1
    assert ctx.proposals[0][0].side == "short"


@pytest.mark.asyncio
async def test_htf_break_rejects_when_prior_close_already_above():
    """If the prior trigger close was already above the level, we don't have
    a fresh break — could be ongoing trend or a gap. Skip."""
    strat, ctx = _make_strategy(trendline_enabled=False)
    await strat.start(ctx)
    htf_k = _kline("BTCUSDT", "1d", 1_000_000, 100, 110, 95, 105)
    await strat.on_bar(htf_k, _snap("BTCUSDT", "1d", 105, atr=2, vol_z=0, rsi=50))
    ctx.indicators.get("BTCUSDT", "1d").last_snapshot = IndicatorSnapshot(  # type: ignore[assignment]
        symbol="BTCUSDT", timeframe="1d", close=120, ema55=100, supertrend_dir=1,
    )

    tf = "5m"
    # Prior bar already closed above the level (gap-up case).
    k1 = _kline("BTCUSDT", tf, 1_010_000, 111, 112, 111, 111.5)
    await strat.on_bar(k1, _snap("BTCUSDT", tf, 111.5, atr=0.5, vol_z=1.5, rsi=60))
    k2 = _kline("BTCUSDT", tf, 1_010_300, 112, 113, 111.5, 112.5)
    await strat.on_bar(k2, _snap("BTCUSDT", tf, 112.5, atr=0.5, vol_z=1.5, rsi=62))
    assert ctx.proposals == []


@pytest.mark.asyncio
async def test_htf_disagreement_blocks_long():
    """HTF supertrend down + close below ema55 should veto a long break."""
    strat, ctx = _make_strategy(trendline_enabled=False)
    await strat.start(ctx)
    htf_k = _kline("BTCUSDT", "1d", 1_000_000, 100, 110, 95, 105)
    await strat.on_bar(htf_k, _snap("BTCUSDT", "1d", 105, atr=2, vol_z=0, rsi=50))
    # HTF regime disagrees: close below ema55, supertrend down.
    ctx.indicators.get("BTCUSDT", "1d").last_snapshot = IndicatorSnapshot(  # type: ignore[assignment]
        symbol="BTCUSDT", timeframe="1d", close=90, ema55=100, supertrend_dir=-1,
    )

    tf = "5m"
    k1 = _kline("BTCUSDT", tf, 1_010_000, 108, 109, 107, 108)
    await strat.on_bar(k1, _snap("BTCUSDT", tf, 108, atr=0.5, vol_z=1.2, rsi=55))
    k2 = _kline("BTCUSDT", tf, 1_010_300, 109, 112, 109, 111)
    await strat.on_bar(k2, _snap("BTCUSDT", tf, 111, atr=0.5, vol_z=1.5, rsi=58))
    assert ctx.proposals == []


@pytest.mark.asyncio
async def test_low_volume_blocks_signal():
    strat, ctx = _make_strategy(trendline_enabled=False, vol_z_min=1.0)
    await strat.start(ctx)
    htf_k = _kline("BTCUSDT", "1d", 1_000_000, 100, 110, 95, 105)
    await strat.on_bar(htf_k, _snap("BTCUSDT", "1d", 105, atr=2, vol_z=0, rsi=50))
    ctx.indicators.get("BTCUSDT", "1d").last_snapshot = IndicatorSnapshot(  # type: ignore[assignment]
        symbol="BTCUSDT", timeframe="1d", close=120, ema55=100, supertrend_dir=1,
    )

    tf = "5m"
    k1 = _kline("BTCUSDT", tf, 1_010_000, 108, 109, 107, 108)
    await strat.on_bar(k1, _snap("BTCUSDT", tf, 108, atr=0.5, vol_z=0.1, rsi=55))
    k2 = _kline("BTCUSDT", tf, 1_010_300, 109, 112, 109, 111)
    # vol_z=0.2 below the 1.0 threshold → reject
    await strat.on_bar(k2, _snap("BTCUSDT", tf, 111, atr=0.5, vol_z=0.2, rsi=58))
    assert ctx.proposals == []


@pytest.mark.asyncio
async def test_cooldown_blocks_immediate_re_fire():
    """After a signal fires, an immediate second candidate (new HTF level,
    fresh break, regime still up) must be blocked by the cooldown. We use
    a long cooldown (240m) so wall-clock time during the test stays inside
    the window."""
    strat, ctx = _make_strategy(trendline_enabled=False, cooldown_min=240)
    await strat.start(ctx)

    # First HTF level + break → signal 1.
    htf1 = _kline("BTCUSDT", "1d", 1_000_000, 100, 110, 95, 105)
    await strat.on_bar(htf1, _snap("BTCUSDT", "1d", 105, atr=2, vol_z=0, rsi=50))
    ctx.indicators.get("BTCUSDT", "1d").last_snapshot = IndicatorSnapshot(
        symbol="BTCUSDT", timeframe="1d", close=120, ema55=100, supertrend_dir=1,
    )
    tf = "5m"
    k1 = _kline("BTCUSDT", tf, 1_010_000, 108, 109, 107, 108)
    await strat.on_bar(k1, _snap("BTCUSDT", tf, 108, atr=0.5, vol_z=1.2, rsi=55))
    k2 = _kline("BTCUSDT", tf, 1_010_300, 109, 112, 109, 111)
    await strat.on_bar(k2, _snap("BTCUSDT", tf, 111, atr=0.5, vol_z=1.5, rsi=58))
    assert len(ctx.proposals) == 1

    # A *new* level (higher) + fresh break would normally trigger again —
    # but cooldown is still active (wall-clock has barely moved).
    htf2 = _kline("BTCUSDT", "1d", 1_100_000, 110, 120, 108, 115)
    await strat.on_bar(htf2, _snap("BTCUSDT", "1d", 115, atr=2, vol_z=0, rsi=50))
    k3 = _kline("BTCUSDT", tf, 1_110_000, 118, 119, 118, 118.5)
    await strat.on_bar(k3, _snap("BTCUSDT", tf, 118.5, atr=0.5, vol_z=1.2, rsi=55))
    k4 = _kline("BTCUSDT", tf, 1_110_300, 119, 122, 119, 121)
    await strat.on_bar(k4, _snap("BTCUSDT", tf, 121, atr=0.5, vol_z=1.5, rsi=58))
    assert len(ctx.proposals) == 1, "cooldown should still be active"


@pytest.mark.asyncio
async def test_zero_cooldown_allows_re_fire():
    """Sanity check: when cooldown is 0, a fresh break on a NEW level
    fires a second signal. This guards against accidentally blocking
    on something other than cooldown."""
    strat, ctx = _make_strategy(trendline_enabled=False, cooldown_min=0)
    await strat.start(ctx)

    htf1 = _kline("BTCUSDT", "1d", 1_000_000, 100, 110, 95, 105)
    await strat.on_bar(htf1, _snap("BTCUSDT", "1d", 105, atr=2, vol_z=0, rsi=50))
    ctx.indicators.get("BTCUSDT", "1d").last_snapshot = IndicatorSnapshot(
        symbol="BTCUSDT", timeframe="1d", close=120, ema55=100, supertrend_dir=1,
    )
    tf = "5m"
    await strat.on_bar(
        _kline("BTCUSDT", tf, 1_010_000, 108, 109, 107, 108),
        _snap("BTCUSDT", tf, 108, atr=0.5, vol_z=1.2, rsi=55),
    )
    await strat.on_bar(
        _kline("BTCUSDT", tf, 1_010_300, 109, 112, 109, 111),
        _snap("BTCUSDT", tf, 111, atr=0.5, vol_z=1.5, rsi=58),
    )
    assert len(ctx.proposals) == 1

    # New higher level + fresh break.
    htf2 = _kline("BTCUSDT", "1d", 1_100_000, 110, 120, 108, 115)
    await strat.on_bar(htf2, _snap("BTCUSDT", "1d", 115, atr=2, vol_z=0, rsi=50))
    await strat.on_bar(
        _kline("BTCUSDT", tf, 1_110_000, 118, 119, 118, 118.5),
        _snap("BTCUSDT", tf, 118.5, atr=0.5, vol_z=1.2, rsi=55),
    )
    await strat.on_bar(
        _kline("BTCUSDT", tf, 1_110_300, 119, 122, 119, 121),
        _snap("BTCUSDT", tf, 121, atr=0.5, vol_z=1.5, rsi=58),
    )
    assert len(ctx.proposals) == 2


# ─────────────────────────────── trendline path ────────────────────────


@pytest.mark.asyncio
async def test_trendline_long_break_fires():
    """Build two falling pivot highs, then have price break above the line."""
    strat, ctx = _make_strategy(
        trigger_tf="15m", trendline_tf="15m", pivot_window=2,
    )
    await strat.start(ctx)
    # No HTF break level set (would mask trendline path). Don't feed an HTF bar.
    ctx.indicators.get("BTCUSDT", "1d").last_snapshot = IndicatorSnapshot(  # type: ignore[assignment]
        symbol="BTCUSDT", timeframe="1d", close=120, ema55=100, supertrend_dir=1,
    )

    tf = "15m"

    def push(ts: int, h: float, l: float, c: float):
        return strat.on_bar(
            _kline("BTCUSDT", tf, ts, c - 1, h, l, c),
            _snap("BTCUSDT", tf, c, atr=0.5, vol_z=1.5, rsi=55),
        )

    # Two pivot highs: pivot1 at bar 3 (price 120), pivot2 at bar 9 (price 115).
    # Slope < 0 → falling resistance.
    # Layout (h,l,c):
    #   bars 0..2: build-up
    #   bar 3: PIVOT HIGH 120
    #   bars 4..5: confirm pivot 3
    #   bars 6..8: build-up
    #   bar 9: PIVOT HIGH 115
    #   bars 10..11: confirm pivot 9
    #   bar 12: project line ≈ 113.something; close > line breaks long
    layout = [
        # bar 0..2
        (100, 102, 98, 101),
        (101, 104, 100, 103),
        (103, 106, 102, 105),
        # bar 3 — pivot high 120
        (105, 120, 104, 110),
        # bar 4..5 confirm
        (110, 112, 108, 109),
        (109, 110, 107, 108),
        # bar 6..8 build
        (108, 110, 106, 109),
        (109, 112, 108, 111),
        (111, 114, 110, 113),
        # bar 9 — pivot high 115
        (113, 115, 112, 114),
        # bar 10..11 confirm
        (114, 113, 110, 111),  # high 113 < 115 → confirms pivot 9
        (111, 112, 109, 110),
        # bar 12 — line value between p1(ix=3,price=120) and p2(ix=9,price=115)
        # slope = (115-120)/(9-3) = -0.833. At ix=12: 115 + (-0.833)*(12-9) = 112.5
        # Close > 112.5 to break.
        (110, 116, 109, 115),
    ]
    ts0 = 1_000_000
    for i, (o, h, l, c) in enumerate(layout):
        _ = o  # open is computed inside _kline by c-1; o unused
        await push(ts0 + i * 60_000, h, l, c)

    assert any(p[0].side == "long" and "trendline" in p[0].rationale
               for p in ctx.proposals), \
        f"expected a trendline long; got {[(p[0].side, p[0].rationale) for p in ctx.proposals]}"


@pytest.mark.asyncio
async def test_trendline_requires_falling_slope_for_long():
    """If the two pivot highs slope UP (higher highs), there's no resistance
    line to break — that's a trend, not a setup. Should NOT fire."""
    strat, ctx = _make_strategy(
        trigger_tf="15m", trendline_tf="15m", pivot_window=2,
    )
    await strat.start(ctx)
    ctx.indicators.get("BTCUSDT", "1d").last_snapshot = IndicatorSnapshot(  # type: ignore[assignment]
        symbol="BTCUSDT", timeframe="1d", close=120, ema55=100, supertrend_dir=1,
    )

    tf = "15m"
    # Pivot1 at bar 3 = 110, pivot2 at bar 9 = 120 (rising). Slope > 0.
    layout = [
        (100, 102, 98, 101),
        (101, 104, 100, 103),
        (103, 106, 102, 105),
        (105, 110, 104, 108),   # pivot high 110 (lower than later)
        (108, 109, 106, 107),
        (107, 108, 105, 106),
        (106, 108, 105, 107),
        (107, 110, 106, 109),
        (109, 113, 108, 111),
        (111, 120, 110, 115),   # pivot high 120
        (115, 118, 113, 114),
        (114, 116, 112, 113),
        (113, 122, 112, 121),   # breaks above the (rising) line — should NOT fire
    ]
    ts0 = 1_000_000
    for i, (_, h, l, c) in enumerate(layout):
        await strat.on_bar(
            _kline("BTCUSDT", tf, ts0 + i * 60_000, c - 1, h, l, c),
            _snap("BTCUSDT", tf, c, atr=0.5, vol_z=1.5, rsi=55),
        )

    tl_long = [p for p in ctx.proposals
               if p[0].side == "long" and "trendline" in p[0].rationale]
    assert tl_long == []
