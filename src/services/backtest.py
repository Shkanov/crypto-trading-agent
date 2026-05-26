"""Walk-forward backtester for both strategies.

The point: get an honest historical estimate of expected P&L before risking
even paper capital on extended live runs. Goal isn't to PROVE edge — it's
to FALSIFY it. Most strategies fail here, which is the right outcome.

Two modes:

1. **Indicator-confluence**: replay kline history through the same signal
   generator + risk gate, simulate fills at next-bar open with realistic
   slippage and fees, manage stop/TP via the same PositionManager logic.

2. **Funding-harvest**: replay historical funding rates + spot/perp price
   series, enter pairs on the same thresholds, accrue funding per 8h cycle,
   close on funding flips / basis blowups. Subtract realistic trading costs
   on every open/close.

Outputs: per-strategy summary (n trades, win rate, total return, max DD,
Sharpe, deflated Sharpe with parameter-count penalty) + per-trade ledger.

Backtesting limitations to remember:
- Historical klines are EVENT-DRIVEN survivorship-biased (no delisted alts).
- aggTrade-derived CVD is approximate vs real order-flow CVD.
- Slippage model is constant-bps; real slippage spikes in volatile bars.
- Funding REPLAYS the EXACT historical rate at the EXACT funding time, but
  you wouldn't have known it 8h in advance. The strategy uses 21-period
  trailing average, which IS knowable in real time, so this is OK.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import structlog

from src.config.settings import Settings, get_settings
from src.models.types import IndicatorSnapshot, Kline, Signal, StrategyConfig
from src.strategies.cascade_breakout import (
    CascadeParams,
    CascadePattern,
    _atr,
    detect_pattern,
)
from src.strategies.level_breakout import (
    LevelBreakoutParams,
    _PivotBuffer,
    _build_signal,
    _htf_regime_long_ok,
    _htf_regime_short_ok,
    _trendline_slope,
    _trendline_value_at,
    _trigger_filters_ok,
)
from src.strategies.mean_reversion import (
    MeanReversionConfig,
    generate_mean_reversion_signal,
)
from src.tools.binance_client import BinanceClient
from src.tools.indicators import IndicatorEngine
from src.tools.signal_generator import generate_signal

log = structlog.get_logger(__name__)


@dataclass
class SimTrade:
    symbol: str
    strategy: str
    side: str               # "long" | "short"
    qty: float
    entry_price: float
    stop: float
    tp: float
    entry_ts_ms: int
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_ts_ms: Optional[int] = None
    pnl_usd: Optional[float] = None


@dataclass
class BacktestStats:
    strategy: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_usd: float = 0.0
    avg_pnl_usd: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    deflated_sharpe: float = 0.0
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    starting_equity_usd: float = 0.0
    ending_equity_usd: float = 0.0
    annualized_pct: float = 0.0


def _equity_curve(pnls: list[float], start_equity: float) -> np.ndarray:
    return np.cumsum(np.array([start_equity] + pnls))


def _max_drawdown(curve: np.ndarray) -> tuple[float, float]:
    if curve.size == 0:
        return 0.0, 0.0
    peaks = np.maximum.accumulate(curve)
    dd = curve - peaks
    pct = dd / peaks
    return float(-dd.min()), float(-pct.min() * 100.0)


def _sharpe(pnls: list[float], periods_per_year: float = 365.0) -> float:
    """Per-period Sharpe annualized by sqrt(periods_per_year). The caller is
    responsible for passing a HONEST `periods_per_year` — e.g. for a strategy
    making ~1 trade/week, this should be 52, not 365. Defaulting to 365 was
    a known overstatement bug — kept here for callers that explicitly pass
    a corrected value via `_sharpe_from_pnls_and_span` below."""
    if len(pnls) < 2:
        return 0.0
    a = np.array(pnls)
    if a.std(ddof=1) == 0:
        return 0.0
    return float((a.mean() / a.std(ddof=1)) * math.sqrt(periods_per_year))


def _sharpe_from_pnls_and_span(pnls: list[float], span_days: float) -> float:
    """Annualized Sharpe using ACTUAL realized trade frequency.

    For N trades over `span_days`, periods_per_year = N * (365 / span_days).
    Then Sharpe = (mean / std) * sqrt(periods_per_year).

    This is the honest version. A strategy making 1 trade/week → ~52
    periods/year; not 365. The old default massively overstated Sharpe."""
    if len(pnls) < 2 or span_days <= 0:
        return 0.0
    periods_per_year = len(pnls) * (365.0 / span_days)
    return _sharpe(pnls, periods_per_year=periods_per_year)


def _deflated_sharpe(sharpe: float, n_trades: int, n_trials: int = 5) -> float:
    """López de Prado deflation. Without the full skew/kurt machinery we use a
    conservative approximation: Sharpe is penalized by `sqrt(log(n_trials))/n`."""
    if n_trades < 2:
        return 0.0
    penalty = math.sqrt(math.log(max(2, n_trials))) / math.sqrt(n_trades)
    return max(0.0, sharpe - penalty)


def _stats_from_trades(strategy: str, trades: list[SimTrade],
                       start_equity: float, span_days: float) -> BacktestStats:
    pnls = [t.pnl_usd for t in trades if t.pnl_usd is not None]
    out = BacktestStats(strategy=strategy, starting_equity_usd=start_equity)
    if not pnls:
        out.ending_equity_usd = start_equity
        return out
    out.trades = len(pnls)
    out.wins = sum(1 for p in pnls if p > 0)
    out.losses = sum(1 for p in pnls if p < 0)
    out.total_pnl_usd = sum(pnls)
    out.avg_pnl_usd = out.total_pnl_usd / out.trades
    out.win_rate = out.wins / out.trades
    # F9: annualize using realized trade frequency, not a flat 365.
    out.sharpe = _sharpe_from_pnls_and_span(pnls, span_days)
    out.deflated_sharpe = _deflated_sharpe(out.sharpe, out.trades)
    curve = _equity_curve(pnls, start_equity)
    out.ending_equity_usd = float(curve[-1])
    out.max_drawdown_usd, out.max_drawdown_pct = _max_drawdown(curve)
    if span_days > 0 and start_equity > 0:
        out.annualized_pct = (out.total_pnl_usd / start_equity) * (365.0 / span_days) * 100.0
    return out


# ─────────────────────────────  Indicator strategy backtest  ─────────────────


async def backtest_indicator(
    binance: BinanceClient,
    symbol: str,
    tf: str = "5m",
    htf: str = "1h",
    bars: int = 5000,
    cfg: Optional[StrategyConfig] = None,
    settings: Optional[Settings] = None,
) -> tuple[BacktestStats, list[SimTrade]]:
    s = settings or get_settings()
    cfg = cfg or StrategyConfig(allowed_symbols=[symbol], htf_timeframe=htf)
    ind = IndicatorEngine()

    raw = await binance.fetch_klines_paginated(symbol, tf, total=bars, market="spot")
    ks = [Kline(
        symbol=symbol, timeframe=tf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in raw]
    htf_raw = await binance.fetch_klines_paginated(symbol, htf,
                                                    total=max(300, bars // 12),
                                                    market="spot")
    htf_ks = [Kline(
        symbol=symbol, timeframe=htf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in htf_raw]
    # Warmup with first 200 bars; then iterate forward bar-by-bar.
    warmup_n = min(200, len(ks) // 4)
    ind.warmup(symbol, tf, ks[:warmup_n])
    ind.warmup(symbol, htf, htf_ks)

    trades: list[SimTrade] = []
    open_trade: Optional[SimTrade] = None
    equity = s.account_equity_usd
    fee_bps = s.spot_taker_fee_bps + s.slippage_bps  # one-way

    stop_slip = s.paper_stop_slippage_bps / 10_000
    tp_slip = s.paper_tp_slippage_bps / 10_000

    for k in ks[warmup_n:]:
        # Resolve any open trade against THIS bar first (price might have hit stop/TP).
        if open_trade is not None:
            hit_stop = (k.low <= open_trade.stop) if open_trade.side == "long" else (k.high >= open_trade.stop)
            hit_tp = (k.high >= open_trade.tp) if open_trade.side == "long" else (k.low <= open_trade.tp)
            if hit_stop:
                # Adverse slippage on stop, from settings (paper_stop_slippage_bps).
                exit_px = open_trade.stop * (1 - stop_slip) if open_trade.side == "long" else open_trade.stop * (1 + stop_slip)
                open_trade.exit_price = exit_px
                open_trade.exit_reason = "stop"
                open_trade.exit_ts_ms = k.close_time
                gross = (exit_px - open_trade.entry_price) * open_trade.qty
                if open_trade.side == "short":
                    gross = -gross
                # Round-trip fees + entry slippage already in entry_price.
                exit_fee = open_trade.qty * exit_px * (fee_bps / 10_000)
                open_trade.pnl_usd = gross - exit_fee
                trades.append(open_trade)
                open_trade = None
            elif hit_tp:
                exit_px = open_trade.tp * (1 - tp_slip) if open_trade.side == "long" else open_trade.tp * (1 + tp_slip)
                open_trade.exit_price = exit_px
                open_trade.exit_reason = "tp"
                open_trade.exit_ts_ms = k.close_time
                gross = (exit_px - open_trade.entry_price) * open_trade.qty
                if open_trade.side == "short":
                    gross = -gross
                exit_fee = open_trade.qty * exit_px * (fee_bps / 10_000)
                open_trade.pnl_usd = gross - exit_fee
                trades.append(open_trade)
                open_trade = None

        # Update indicators with this closed bar.
        snap = ind.get(symbol, tf).on_closed_kline(k)

        # No new entries while a trade is open (one-at-a-time per symbol).
        if open_trade is not None:
            continue

        htf_snap = ind.latest(symbol, htf)
        sig: Optional[Signal] = generate_signal(symbol, snap, htf_snap, cfg)
        if not sig:
            continue
        # Cost-of-edge filter is built into generate_signal already.
        risk_pct = s.risk_per_trade_pct / 100.0
        risk_usd = equity * risk_pct
        risk_per_unit = abs(sig.entry - sig.stop)
        if risk_per_unit <= 0:
            continue
        qty = min(risk_usd / risk_per_unit, s.max_notional_usd / sig.entry)
        # Entry slippage built into entry_price (worse for buyer).
        entry_slip = sig.entry * (s.slippage_bps / 10_000)
        entry_px = sig.entry + entry_slip if sig.side == "long" else sig.entry - entry_slip
        open_trade = SimTrade(
            symbol=symbol, strategy="indicator",
            side=sig.side, qty=qty,
            entry_price=entry_px, stop=sig.stop, tp=sig.take_profit,
            entry_ts_ms=k.close_time,
        )

    # Close any dangling trade at the last bar's close (no-edge exit).
    if open_trade is not None and ks:
        last = ks[-1].close
        open_trade.exit_price = last
        open_trade.exit_reason = "eod"
        open_trade.exit_ts_ms = ks[-1].close_time
        gross = (last - open_trade.entry_price) * open_trade.qty
        if open_trade.side == "short":
            gross = -gross
        open_trade.pnl_usd = gross
        trades.append(open_trade)

    span_days = (ks[-1].close_time - ks[0].close_time) / 1000 / 86400 if ks else 0.0
    return _stats_from_trades("indicator", trades, s.account_equity_usd, span_days), trades


# ─────────────────────────────  Mean-reversion strategy backtest  ────────────


async def backtest_mean_reversion(
    binance: BinanceClient,
    symbol: str,
    tf: str = "5m",
    htf: str = "1h",
    bars: int = 5000,
    cfg: Optional[MeanReversionConfig] = None,
    settings: Optional[Settings] = None,
) -> tuple[BacktestStats, list[SimTrade]]:
    """Walk-forward replay of the mean-reversion strategy.

    Same fill / slippage / fee model as `backtest_indicator` — different
    only in how each bar's Signal is generated. This guarantees comparable
    P&L numbers between the two strategies on the same symbol/window.
    """
    s = settings or get_settings()
    cfg = cfg or MeanReversionConfig(allowed_symbols=[symbol], htf_timeframe=htf)
    ind = IndicatorEngine()

    raw = await binance.fetch_klines_paginated(symbol, tf, total=bars, market="spot")
    ks = [Kline(
        symbol=symbol, timeframe=tf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in raw]
    htf_raw = await binance.fetch_klines_paginated(symbol, htf,
                                                    total=max(300, bars // 12),
                                                    market="spot")
    htf_ks = [Kline(
        symbol=symbol, timeframe=htf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in htf_raw]
    warmup_n = min(200, len(ks) // 4)
    ind.warmup(symbol, tf, ks[:warmup_n])
    ind.warmup(symbol, htf, htf_ks)

    trades: list[SimTrade] = []
    open_trade: Optional[SimTrade] = None
    bars_held = 0
    equity = s.account_equity_usd
    fee_bps = s.spot_taker_fee_bps + s.slippage_bps  # one-way
    stop_slip = s.paper_stop_slippage_bps / 10_000
    tp_slip = s.paper_tp_slippage_bps / 10_000

    for k in ks[warmup_n:]:
        if open_trade is not None:
            hit_stop = (k.low <= open_trade.stop) if open_trade.side == "long" else (k.high >= open_trade.stop)
            hit_tp = (k.high >= open_trade.tp) if open_trade.side == "long" else (k.low <= open_trade.tp)
            if hit_stop:
                exit_px = open_trade.stop * (1 - stop_slip) if open_trade.side == "long" else open_trade.stop * (1 + stop_slip)
                open_trade.exit_price = exit_px
                open_trade.exit_reason = "stop"
                open_trade.exit_ts_ms = k.close_time
                gross = (exit_px - open_trade.entry_price) * open_trade.qty
                if open_trade.side == "short":
                    gross = -gross
                exit_fee = open_trade.qty * exit_px * (fee_bps / 10_000)
                open_trade.pnl_usd = gross - exit_fee
                trades.append(open_trade)
                open_trade = None
                bars_held = 0
            elif hit_tp:
                exit_px = open_trade.tp * (1 - tp_slip) if open_trade.side == "long" else open_trade.tp * (1 + tp_slip)
                open_trade.exit_price = exit_px
                open_trade.exit_reason = "tp"
                open_trade.exit_ts_ms = k.close_time
                gross = (exit_px - open_trade.entry_price) * open_trade.qty
                if open_trade.side == "short":
                    gross = -gross
                exit_fee = open_trade.qty * exit_px * (fee_bps / 10_000)
                open_trade.pnl_usd = gross - exit_fee
                trades.append(open_trade)
                open_trade = None
                bars_held = 0
            elif bars_held >= cfg.time_stop_bars:
                # Time-stop: failed revert thesis. Exit at this bar's close.
                # Apply same adverse-slippage assumption as stops, since we're
                # crossing the spread to exit at market, but the move was
                # smaller (no gap), so use tp_slip instead of stop_slip.
                exit_px = k.close * (1 - tp_slip) if open_trade.side == "long" else k.close * (1 + tp_slip)
                open_trade.exit_price = exit_px
                open_trade.exit_reason = "time_stop"
                open_trade.exit_ts_ms = k.close_time
                gross = (exit_px - open_trade.entry_price) * open_trade.qty
                if open_trade.side == "short":
                    gross = -gross
                exit_fee = open_trade.qty * exit_px * (fee_bps / 10_000)
                open_trade.pnl_usd = gross - exit_fee
                trades.append(open_trade)
                open_trade = None
                bars_held = 0
            else:
                bars_held += 1

        snap = ind.get(symbol, tf).on_closed_kline(k)
        if open_trade is not None:
            continue
        htf_snap = ind.latest(symbol, htf)
        sig: Optional[Signal] = generate_mean_reversion_signal(symbol, snap, htf_snap, cfg)
        if not sig:
            continue

        risk_pct = s.risk_per_trade_pct / 100.0
        risk_usd = equity * risk_pct
        risk_per_unit = abs(sig.entry - sig.stop)
        if risk_per_unit <= 0:
            continue
        qty = min(risk_usd / risk_per_unit, s.max_notional_usd / sig.entry)
        entry_slip = sig.entry * (s.slippage_bps / 10_000)
        entry_px = sig.entry + entry_slip if sig.side == "long" else sig.entry - entry_slip
        open_trade = SimTrade(
            symbol=symbol, strategy="mean_reversion",
            side=sig.side, qty=qty,
            entry_price=entry_px, stop=sig.stop, tp=sig.take_profit,
            entry_ts_ms=k.close_time,
        )
        bars_held = 0

    if open_trade is not None and ks:
        last = ks[-1].close
        open_trade.exit_price = last
        open_trade.exit_reason = "eod"
        open_trade.exit_ts_ms = ks[-1].close_time
        gross = (last - open_trade.entry_price) * open_trade.qty
        if open_trade.side == "short":
            gross = -gross
        open_trade.pnl_usd = gross
        trades.append(open_trade)

    span_days = (ks[-1].close_time - ks[0].close_time) / 1000 / 86400 if ks else 0.0
    return _stats_from_trades("mean_reversion", trades, s.account_equity_usd, span_days), trades


# ─────────────────  Scaled mean-reversion (v2) backtest  ────────────────────


@dataclass
class _ScaledMRState:
    """In-flight state of a scaled-mean-rev trade. Tracks multiple entry
    legs, partial TP1 close, and overall realized P&L from partials."""
    symbol: str
    side: str
    legs: list[tuple[int, float, float]] = field(default_factory=list)  # (ts_ms, price, qty)
    initial_entry_price: float = 0.0
    initial_atr: float = 0.0
    tp2_target: float = 0.0
    tp1_hit: bool = False
    closed_qty: float = 0.0
    realized_pnl_usd: float = 0.0
    bars_held: int = 0

    @property
    def total_qty(self) -> float:
        return sum(q for _, _, q in self.legs)

    @property
    def avg_entry(self) -> float:
        tq = self.total_qty
        if tq <= 0:
            return 0.0
        return sum(p * q for _, p, q in self.legs) / tq

    @property
    def open_qty(self) -> float:
        return max(0.0, self.total_qty - self.closed_qty)

    def stop_price(self, mult: float) -> float:
        if self.side == "long":
            return self.avg_entry - mult * self.initial_atr
        return self.avg_entry + mult * self.initial_atr

    def tp1_price(self, distance_frac: float) -> float:
        avg = self.avg_entry
        return avg + (self.tp2_target - avg) * distance_frac


async def backtest_mean_reversion_scaled(
    binance: BinanceClient,
    symbol: str,
    tf: str = "5m",
    htf: str = "1h",
    bars: int = 5000,
    cfg: Optional["ScaledMeanReversionConfig"] = None,
    settings: Optional[Settings] = None,
) -> tuple[BacktestStats, list[SimTrade]]:
    """Scaled mean-rev: scale-in entries + partial TP + wider stop.

    Fixes the broken R/R of mean_reversion v1 (~1:1 R/R, needs 53% win
    rate to clear costs but observed 18-30%). Geometry:

      - Initial entry on standard mean-rev gate (RSI/StochRSI/BB/ADX)
      - Second entry if price moves `scale_in_atr_step` ATR further into
        the stretch AND gates still hold (i.e. still oversold / overbought)
      - Stop: 3 ATR from AVERAGE entry (vs 1.5 in v1) — gives the thesis
        room to breathe
      - TP1: halfway from avg to BB middle — close 50% of position
      - TP2: BB middle — close remaining
      - Time-stop: 48 bars (doubled vs v1)

    Each completed scaled trade emits ONE SimTrade record with the FULL
    realized P&L (TP1 partial + final exit). entry_price is the avg of
    all legs, exit_price is the final exit (or the only exit if no TP1).
    """
    from src.strategies.mean_reversion import ScaledMeanReversionConfig

    s = settings or get_settings()
    cfg = cfg or ScaledMeanReversionConfig(allowed_symbols=[symbol], htf_timeframe=htf)
    ind = IndicatorEngine()

    raw = await binance.fetch_klines_paginated(symbol, tf, total=bars, market="spot")
    ks = [Kline(
        symbol=symbol, timeframe=tf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in raw]
    htf_raw = await binance.fetch_klines_paginated(symbol, htf,
                                                    total=max(300, bars // 12),
                                                    market="spot")
    htf_ks = [Kline(
        symbol=symbol, timeframe=htf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in htf_raw]
    warmup_n = min(200, len(ks) // 4)
    ind.warmup(symbol, tf, ks[:warmup_n])
    ind.warmup(symbol, htf, htf_ks)

    trades: list[SimTrade] = []
    state: Optional[_ScaledMRState] = None
    equity = s.account_equity_usd
    fee_bps = s.spot_taker_fee_bps + s.slippage_bps  # one-way
    stop_slip = s.paper_stop_slippage_bps / 10_000
    tp_slip = s.paper_tp_slippage_bps / 10_000
    entry_slip = s.slippage_bps / 10_000

    def _finalize(state: _ScaledMRState, exit_px: float, exit_reason: str,
                  ts_ms: int) -> SimTrade:
        # Compute remaining-leg P&L and combine with already-realized.
        remaining = state.open_qty
        if state.side == "long":
            gross_remaining = (exit_px - state.avg_entry) * remaining
        else:
            gross_remaining = (state.avg_entry - exit_px) * remaining
        exit_fee = remaining * exit_px * (fee_bps / 10_000)
        final_pnl = state.realized_pnl_usd + gross_remaining - exit_fee
        return SimTrade(
            symbol=state.symbol, strategy="mean_reversion_scaled",
            side=state.side, qty=state.total_qty,
            entry_price=state.avg_entry,
            stop=state.stop_price(cfg.atr_stop_mult),
            tp=state.tp2_target, entry_ts_ms=state.legs[0][0],
            exit_price=exit_px, exit_reason=exit_reason,
            exit_ts_ms=ts_ms, pnl_usd=final_pnl,
        )

    def _try_open(snap: IndicatorSnapshot, htf_snap: Optional[IndicatorSnapshot],
                   k: Kline) -> Optional[_ScaledMRState]:
        # Mirror the gate logic from generate_mean_reversion_signal but
        # build a scaled state instead of a Signal.
        if symbol not in cfg.allowed_symbols:
            return None
        if snap.atr14 is None or snap.atr14 <= 0:
            return None
        if snap.bb_upper is None or snap.bb_lower is None or snap.bb_middle is None:
            return None
        if snap.rsi14 is None or snap.stoch_rsi_k is None:
            return None
        if snap.adx14 is None or snap.adx14 > cfg.adx_max_for_meanrev:
            return None
        if htf_snap is not None and htf_snap.adx14 is not None and htf_snap.adx14 > 30.0:
            return None
        side: Optional[str] = None
        if (k.close < snap.bb_lower
                and snap.stoch_rsi_k < cfg.stoch_oversold
                and snap.rsi14 < cfg.rsi_oversold
                and snap.bb_middle > k.close
                and "long" in cfg.enabled_sides):
            side = "long"
        elif (k.close > snap.bb_upper
                and snap.stoch_rsi_k > cfg.stoch_overbought
                and snap.rsi14 > cfg.rsi_overbought
                and snap.bb_middle < k.close
                and "short" in cfg.enabled_sides):
            side = "short"
        if side is None:
            return None
        if abs(snap.bb_middle - k.close) < cfg.min_target_atr * snap.atr14:
            return None
        # Size FIRST leg using full risk budget at the initial stop distance
        # (initial_atr * atr_stop_mult). Scale-in legs ADD to the position
        # — so a scaled trade ends up risking ~1.5-2x the base budget on
        # the wider stop, which is the design intent (higher conviction =
        # more size). Cap at max_notional_usd across all legs.
        risk_pct = s.risk_per_trade_pct / 100.0
        risk_usd = equity * risk_pct
        risk_per_unit = cfg.atr_stop_mult * snap.atr14
        if risk_per_unit <= 0:
            return None
        first_leg_qty = min(risk_usd / risk_per_unit,
                            (s.max_notional_usd / cfg.max_entries) / k.close)
        if first_leg_qty <= 0:
            return None
        entry_px = k.close * (1 + entry_slip) if side == "long" else k.close * (1 - entry_slip)
        # Entry fee on first leg
        entry_fee = first_leg_qty * entry_px * (fee_bps / 10_000)
        st = _ScaledMRState(
            symbol=symbol, side=side,
            legs=[(k.close_time, entry_px, first_leg_qty)],
            initial_entry_price=k.close, initial_atr=snap.atr14,
            tp2_target=snap.bb_middle,
            realized_pnl_usd=-entry_fee,
        )
        return st

    for k in ks[warmup_n:]:
        snap = ind.get(symbol, tf).on_closed_kline(k)
        htf_snap = ind.latest(symbol, htf)

        if state is not None:
            state.bars_held += 1
            avg = state.avg_entry
            stop = state.stop_price(cfg.atr_stop_mult)
            tp1 = state.tp1_price(cfg.tp1_distance_frac)
            tp2 = state.tp2_target

            # Pessimistic ordering: check stop first (assume worst path).
            hit_stop = (k.low <= stop) if state.side == "long" else (k.high >= stop)
            hit_tp2 = (k.high >= tp2) if state.side == "long" else (k.low <= tp2)
            hit_tp1 = (k.high >= tp1) if state.side == "long" else (k.low <= tp1)

            if hit_stop:
                exit_px = stop * (1 - stop_slip) if state.side == "long" else stop * (1 + stop_slip)
                trades.append(_finalize(state, exit_px, "stop", k.close_time))
                state = None
                continue
            if hit_tp2:
                exit_px = tp2 * (1 - tp_slip) if state.side == "long" else tp2 * (1 + tp_slip)
                trades.append(_finalize(state, exit_px, "tp2", k.close_time))
                state = None
                continue
            if not state.tp1_hit and hit_tp1:
                # Realize partial: close `tp1_close_fraction` of total.
                close_qty = state.total_qty * cfg.tp1_close_fraction
                tp1_px = tp1 * (1 - tp_slip) if state.side == "long" else tp1 * (1 + tp_slip)
                if state.side == "long":
                    gross = (tp1_px - avg) * close_qty
                else:
                    gross = (avg - tp1_px) * close_qty
                exit_fee = close_qty * tp1_px * (fee_bps / 10_000)
                state.realized_pnl_usd += gross - exit_fee
                state.closed_qty += close_qty
                state.tp1_hit = True
                # Don't continue — same bar may also have scale-in or time-stop.

            # Scale-in: only if we haven't taken TP1 yet (after TP1, we're
            # reducing not adding) and price is deeper into stretch.
            if (not state.tp1_hit and len(state.legs) < cfg.max_entries
                    and snap.adx14 is not None and snap.adx14 <= cfg.adx_max_for_meanrev):
                step_px = cfg.scale_in_atr_step * state.initial_atr
                if state.side == "long":
                    trigger = state.initial_entry_price - step_px
                    triggered = (k.low <= trigger)
                else:
                    trigger = state.initial_entry_price + step_px
                    triggered = (k.high >= trigger)
                if triggered:
                    leg_qty = state.legs[0][2]  # same as first leg
                    leg_px = trigger * (1 + entry_slip) if state.side == "long" else trigger * (1 - entry_slip)
                    entry_fee = leg_qty * leg_px * (fee_bps / 10_000)
                    state.realized_pnl_usd -= entry_fee
                    state.legs.append((k.close_time, leg_px, leg_qty))

            if state is not None and state.bars_held >= cfg.time_stop_bars:
                exit_px = k.close * (1 - tp_slip) if state.side == "long" else k.close * (1 + tp_slip)
                trades.append(_finalize(state, exit_px, "time_stop", k.close_time))
                state = None
                continue

        if state is None:
            new_state = _try_open(snap, htf_snap, k)
            if new_state is not None:
                state = new_state

    # Close any dangling state at last bar.
    if state is not None and ks:
        last_k = ks[-1]
        exit_px = last_k.close
        trades.append(_finalize(state, exit_px, "eod", last_k.close_time))

    span_days = (ks[-1].close_time - ks[0].close_time) / 1000 / 86400 if ks else 0.0
    return _stats_from_trades("mean_reversion_scaled", trades,
                              s.account_equity_usd, span_days), trades


# ─────────────────────────────  Level-breakout strategy backtest  ───────────


async def backtest_level_breakout(
    binance: BinanceClient,
    symbol: str,
    tf: str = "15m",
    htf: str = "1d",
    bars: int = 5000,
    params: Optional[LevelBreakoutParams] = None,
    settings: Optional[Settings] = None,
    market: str = "spot",
) -> tuple[BacktestStats, list[SimTrade]]:
    """Walk-forward replay of level-breakout (fetch + simulate).

    `market` controls which Binance endpoint serves klines AND which
    taker-fee constant is used in the cost model. Pass "perps" when
    validating against the channel's universe (it trades perps); the
    default "spot" matches the other backtests in this module for
    apples-to-apples comparison on the same symbol.

    Restriction vs. live: trigger_tf must equal trendline_tf (so we only
    interleave HTF + one trigger series). The live strategy supports a
    third TF for trendlines, but in v1 of the backtest we unify them.
    """
    s = settings or get_settings()
    params = params or LevelBreakoutParams(htf=htf, trigger_tf=tf, trendline_tf=tf)

    bars_per_day = max(1, int(round(1440 / _tf_minutes(tf))))
    htf_total = max(60, bars // bars_per_day + 60)

    raw = await binance.fetch_klines_paginated(symbol, tf, total=bars, market=market)
    ks = [Kline(
        symbol=symbol, timeframe=tf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in raw]

    htf_raw = await binance.fetch_klines_paginated(symbol, htf, total=htf_total,
                                                    market=market)
    htf_ks = [Kline(
        symbol=symbol, timeframe=htf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in htf_raw]

    return simulate_level_breakout(symbol, ks, htf_ks, params=params,
                                    settings=s, market=market)


def simulate_level_breakout(
    symbol: str,
    ks: list[Kline],
    htf_ks: list[Kline],
    *,
    params: Optional[LevelBreakoutParams] = None,
    settings: Optional[Settings] = None,
    market: str = "spot",
) -> tuple[BacktestStats, list[SimTrade]]:
    """Pure-data simulation. Same logic as the fetch+sim entry point, but
    takes pre-fetched klines so callers (walk-forward driver, multi-symbol
    sweep) can fetch once and slice.

    Honest caveats specific to this backtest:
    - Donchian/EMA/etc. on the trigger TF need warmup. We warmup with the
      first 200 trigger bars; signals before that point are skipped.
    - Prior-HTF level updates lag by one HTF bar — correct: the level you
      can trade off of today is yesterday's high/low.
    - Pivots are confirmed with a `pivot_window` lookforward, so by
      construction we never act on a pivot we couldn't have known.
    """
    s = settings or get_settings()
    params = params or LevelBreakoutParams()
    ind = IndicatorEngine()
    tf = params.trigger_tf
    htf = params.htf

    warmup_n = min(200, len(ks) // 4)
    ind.warmup(symbol, tf, ks[:warmup_n])
    ind.warmup(symbol, htf, htf_ks)

    htf_close_times = [hk.close_time for hk in htf_ks]
    htf_highs = [hk.high for hk in htf_ks]
    htf_lows = [hk.low for hk in htf_ks]

    def _prior_htf_high_low(ts_ms: int) -> tuple[Optional[float], Optional[float]]:
        lo, hi = 0, len(htf_close_times)
        while lo < hi:
            mid = (lo + hi) // 2
            if htf_close_times[mid] < ts_ms:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx < 0:
            return None, None
        return htf_highs[idx], htf_lows[idx]

    pivots = _PivotBuffer(window=params.pivot_window) if params.trendline_enabled else None

    trades: list[SimTrade] = []
    open_trade: Optional[SimTrade] = None
    last_trigger_close: Optional[float] = None
    cooldown_until_ms: int = 0
    last_signal_bar_ts_ms: int = 0
    equity = s.account_equity_usd
    taker_bps = s.perps_taker_fee_bps if market == "perps" else s.spot_taker_fee_bps
    fee_bps = taker_bps + s.slippage_bps
    stop_slip = s.paper_stop_slippage_bps / 10_000
    tp_slip = s.paper_tp_slippage_bps / 10_000

    for k in ks[warmup_n:]:
        # Resolve any open trade against THIS bar first.
        if open_trade is not None:
            hit_stop = (k.low <= open_trade.stop) if open_trade.side == "long" else (k.high >= open_trade.stop)
            hit_tp = (k.high >= open_trade.tp) if open_trade.side == "long" else (k.low <= open_trade.tp)
            if hit_stop:
                exit_px = open_trade.stop * (1 - stop_slip) if open_trade.side == "long" else open_trade.stop * (1 + stop_slip)
                open_trade.exit_price = exit_px
                open_trade.exit_reason = "stop"
                open_trade.exit_ts_ms = k.close_time
                gross = (exit_px - open_trade.entry_price) * open_trade.qty
                if open_trade.side == "short":
                    gross = -gross
                exit_fee = open_trade.qty * exit_px * (fee_bps / 10_000)
                open_trade.pnl_usd = gross - exit_fee
                trades.append(open_trade)
                # Post-stop cooldown using the BAR's close_time (this is
                # backtest time, not wall-clock — `now_ms()` would be wrong).
                cooldown_until_ms = k.close_time + params.cooldown_min * 60_000
                open_trade = None
            elif hit_tp:
                exit_px = open_trade.tp * (1 - tp_slip) if open_trade.side == "long" else open_trade.tp * (1 + tp_slip)
                open_trade.exit_price = exit_px
                open_trade.exit_reason = "tp"
                open_trade.exit_ts_ms = k.close_time
                gross = (exit_px - open_trade.entry_price) * open_trade.qty
                if open_trade.side == "short":
                    gross = -gross
                exit_fee = open_trade.qty * exit_px * (fee_bps / 10_000)
                open_trade.pnl_usd = gross - exit_fee
                trades.append(open_trade)
                open_trade = None

        # Update indicators with this closed trigger bar.
        snap = ind.get(symbol, tf).on_closed_kline(k)
        # Update pivot buffer (trendline-TF == trigger-TF in this backtest).
        if pivots is not None:
            pivots.push(k.high, k.low)

        # No new entries while a trade is open, or in cooldown.
        if open_trade is not None or k.close_time < cooldown_until_ms:
            last_trigger_close = k.close
            continue
        if last_signal_bar_ts_ms == k.close_time:
            last_trigger_close = k.close
            continue

        # Resolve prior HTF level for this trigger bar's timestamp.
        prior_high, prior_low = _prior_htf_high_low(k.close_time)
        htf_snap = ind.latest(symbol, htf)

        sig: Optional[Signal] = None
        rationale = ""

        # HTF-break path
        if (prior_high is not None and last_trigger_close is not None
                and k.close > prior_high and last_trigger_close <= prior_high
                and _htf_regime_long_ok(htf_snap)):
            ok, _ = _trigger_filters_ok("long", snap, params)
            if ok:
                sig = _build_signal(
                    symbol=symbol, side="long",
                    entry=k.close, atr=snap.atr14 or 0.0, params=params,
                    confidence=params.htf_break_confidence,
                    score=params.htf_break_score,
                    rationale=f"htf_break_long {prior_high:.6f}",
                )
        if (sig is None and prior_low is not None and last_trigger_close is not None
                and k.close < prior_low and last_trigger_close >= prior_low
                and _htf_regime_short_ok(htf_snap)):
            ok, _ = _trigger_filters_ok("short", snap, params)
            if ok:
                sig = _build_signal(
                    symbol=symbol, side="short",
                    entry=k.close, atr=snap.atr14 or 0.0, params=params,
                    confidence=params.htf_break_confidence,
                    score=params.htf_break_score,
                    rationale=f"htf_break_short {prior_low:.6f}",
                )

        # Trendline path (only if HTF-break didn't fire)
        if sig is None and pivots is not None:
            cur_ix = pivots.current_bar_ix
            max_age = params.trendline_max_age_bars
            if (len(pivots.highs) >= 2 and _htf_regime_long_ok(htf_snap)):
                p2, p1 = pivots.highs[-1], pivots.highs[-2]
                if cur_ix - p2.bar_ix <= max_age and _trendline_slope(p1, p2) < 0:
                    line_val = _trendline_value_at(p1, p2, cur_ix)
                    if k.close > line_val:
                        ok, _ = _trigger_filters_ok("long", snap, params)
                        if ok:
                            sig = _build_signal(
                                symbol=symbol, side="long",
                                entry=k.close, atr=snap.atr14 or 0.0,
                                params=params,
                                confidence=params.trendline_confidence,
                                score=params.trendline_score,
                                rationale=f"trendline_long {line_val:.6f}",
                            )
            if sig is None and len(pivots.lows) >= 2 and _htf_regime_short_ok(htf_snap):
                p2, p1 = pivots.lows[-1], pivots.lows[-2]
                if cur_ix - p2.bar_ix <= max_age and _trendline_slope(p1, p2) > 0:
                    line_val = _trendline_value_at(p1, p2, cur_ix)
                    if k.close < line_val:
                        ok, _ = _trigger_filters_ok("short", snap, params)
                        if ok:
                            sig = _build_signal(
                                symbol=symbol, side="short",
                                entry=k.close, atr=snap.atr14 or 0.0,
                                params=params,
                                confidence=params.trendline_confidence,
                                score=params.trendline_score,
                                rationale=f"trendline_short {line_val:.6f}",
                            )

        last_trigger_close = k.close
        if sig is None:
            continue

        # Size, open, mark cooldown so the next bar can't re-fire.
        risk_pct = s.risk_per_trade_pct / 100.0
        risk_usd = equity * risk_pct
        risk_per_unit = abs(sig.entry - sig.stop)
        if risk_per_unit <= 0:
            continue
        qty = min(risk_usd / risk_per_unit, s.max_notional_usd / sig.entry)
        entry_slip = sig.entry * (s.slippage_bps / 10_000)
        entry_px = sig.entry + entry_slip if sig.side == "long" else sig.entry - entry_slip
        open_trade = SimTrade(
            symbol=symbol, strategy="levelbreak",
            side=sig.side, qty=qty,
            entry_price=entry_px, stop=sig.stop, tp=sig.take_profit,
            entry_ts_ms=k.close_time,
        )
        last_signal_bar_ts_ms = k.close_time
        cooldown_until_ms = k.close_time + params.cooldown_min * 60_000

    if open_trade is not None and ks:
        last = ks[-1].close
        open_trade.exit_price = last
        open_trade.exit_reason = "eod"
        open_trade.exit_ts_ms = ks[-1].close_time
        gross = (last - open_trade.entry_price) * open_trade.qty
        if open_trade.side == "short":
            gross = -gross
        open_trade.pnl_usd = gross
        trades.append(open_trade)

    span_days = (ks[-1].close_time - ks[0].close_time) / 1000 / 86400 if ks else 0.0
    return _stats_from_trades("levelbreak", trades, s.account_equity_usd, span_days), trades


def _tf_minutes(tf: str) -> int:
    n = "".join(c for c in tf if c.isdigit()) or "1"
    unit = tf[-1]
    mult = {"m": 1, "h": 60, "d": 1440}.get(unit, 1)
    return int(n) * mult


# ─────────────────────────────  Funding strategy backtest  ──────────────────


async def backtest_funding(
    binance: BinanceClient,
    symbol: str,
    days: int = 30,
    params: Optional["FundingBacktestParams"] = None,
    settings: Optional[Settings] = None,
) -> tuple[BacktestStats, list[SimTrade]]:
    """Replay historical funding payments + spot/perp prices through the same
    decision logic as the live FundingHarvestStrategy.

    Parity with live (F3 fix):
      - Same entry threshold + 21-period avg gate
      - Same fees (spot fee on spot leg, perp fee on perp leg)
      - Slippage on both open and close
      - Basis simulated from premiumIndex history; basis-breakout exit
      - Perp adverse-move stop
    """
    s = settings or get_settings()
    p = params or FundingBacktestParams()
    assert binance.client is not None

    # Historical funding rates (one per 8h).
    # Binance's futures_funding_rate endpoint silently caps at 200 rows when no
    # startTime is supplied. Pass startTime/endTime and page until we've covered
    # the requested window (Binance caps each page at 1000 rows ≈ 333 days).
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400 * 1000
    funding_rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        page = await binance.client.futures_funding_rate(
            symbol=symbol, startTime=cursor, endTime=end_ms, limit=1000,
        )
        if not page:
            break
        funding_rows.extend(page)
        last_t = int(page[-1]["fundingTime"])
        if len(page) < 1000 or last_t <= cursor:
            break
        cursor = last_t + 1
    if not funding_rows:
        return BacktestStats(strategy="funding_harvest"), []
    funding_rows.sort(key=lambda r: int(r["fundingTime"]))
    # Dedupe defensively across page boundaries.
    seen: set[int] = set()
    deduped: list[dict] = []
    for row in funding_rows:
        t = int(row["fundingTime"])
        if t in seen:
            continue
        seen.add(t)
        deduped.append(row)
    funding_rows = deduped

    # 8h klines for spot price reference at each funding time
    spot_raw = await binance.fetch_klines(symbol, "8h", limit=min(1000, days * 3),
                                          market="spot")
    spot_by_t: dict[int, float] = {int(r[6]): float(r[4]) for r in spot_raw}
    # Perp klines for the basis simulation
    try:
        perp_raw = await binance.fetch_klines(symbol, "8h", limit=min(1000, days * 3),
                                              market="perps")
        perp_by_t: dict[int, float] = {int(r[6]): float(r[4]) for r in perp_raw}
    except Exception:
        perp_by_t = {}

    def lookup_px(by_t: dict[int, float], t: int) -> Optional[float]:
        if not by_t:
            return None
        ref_t = min(by_t.keys(), key=lambda x: abs(x - t))
        v = by_t.get(ref_t, 0.0)
        return v if v > 0 else None

    in_pair = False
    direction = 0           # +1 long_spot/short_perp, -1 short_spot/long_perp
    spot_qty = 0.0
    perp_qty = 0.0
    entry_spot_px = 0.0
    entry_perp_px = 0.0
    entry_ts_ms = 0
    accrued_funding_usd = 0.0

    spot_fee_one_way = s.spot_taker_fee_bps / 10_000
    perp_fee_one_way = s.perps_taker_fee_bps / 10_000
    slip_one_way = s.slippage_bps / 10_000
    borrow_8h = p.spot_borrow_bps_per_8h / 10_000

    trades: list[SimTrade] = []
    recent_rates: list[float] = []

    for row in funding_rows:
        rate = float(row["fundingRate"])
        bps = rate * 10_000
        t = int(row["fundingTime"])
        recent_rates.append(rate)
        if len(recent_rates) > 21:
            recent_rates.pop(0)
        avg_bps = (sum(recent_rates) / len(recent_rates)) * 10_000

        spot_px = lookup_px(spot_by_t, t)
        perp_px = lookup_px(perp_by_t, t) or spot_px
        if not spot_px or not perp_px:
            continue
        basis_bps = ((perp_px - spot_px) / spot_px) * 10_000

        if in_pair:
            # Funding accrual — direction-aware
            base = perp_qty * perp_px * rate
            if direction == 1:
                # short perp receives +base
                accrued_funding_usd += base
            else:
                # long perp pays base (i.e. receives -base when rate>0;
                # in a negative-funding regime, rate<0 → -base > 0, i.e. collect)
                accrued_funding_usd += -base
                # Plus spot-borrow cost on short_spot leg
                accrued_funding_usd -= spot_qty * spot_px * borrow_8h

            # Exit logic — direction-aware
            exit_reason: Optional[str] = None
            if direction == 1 and bps <= p.exit_threshold_bps:
                exit_reason = "funding_flip"
            elif direction == -1 and bps >= -p.exit_threshold_bps:
                exit_reason = "funding_flip"
            elif abs(basis_bps) > p.basis_exit_alert_bps:
                exit_reason = "basis_breakout"
            else:
                move_pct = ((perp_px - entry_perp_px) / entry_perp_px) * 100.0
                adverse_pct = move_pct if direction == 1 else -move_pct
                if adverse_pct >= p.perp_adverse_move_pct:
                    exit_reason = "perp_adverse_move"

            if exit_reason is not None:
                # Close side adverse slippage:
                #   +1: SELL spot (long→close = SELL, fills lower), BUY-to-close short perp (fills higher)
                #   -1: BUY spot (short→close = BUY, fills higher), SELL-to-close long perp (fills lower)
                if direction == 1:
                    spot_exit = spot_px * (1 - slip_one_way)
                    perp_exit = perp_px * (1 + slip_one_way)
                    spot_pnl = (spot_exit - entry_spot_px) * spot_qty
                    perp_pnl = (entry_perp_px - perp_exit) * perp_qty   # short profits when exit < entry
                else:
                    spot_exit = spot_px * (1 + slip_one_way)
                    perp_exit = perp_px * (1 - slip_one_way)
                    spot_pnl = (entry_spot_px - spot_exit) * spot_qty   # short_spot
                    perp_pnl = (perp_exit - entry_perp_px) * perp_qty   # long_perp
                spot_fee = spot_qty * spot_exit * spot_fee_one_way
                perp_fee = perp_qty * perp_exit * perp_fee_one_way
                total = accrued_funding_usd + spot_pnl + perp_pnl - spot_fee - perp_fee
                trades.append(SimTrade(
                    symbol=symbol, strategy="funding_harvest",
                    side=("neutral+" if direction == 1 else "neutral-"),
                    qty=spot_qty, entry_price=entry_spot_px,
                    stop=0.0, tp=0.0, entry_ts_ms=entry_ts_ms,
                    exit_price=spot_exit, exit_reason=exit_reason,
                    exit_ts_ms=t, pnl_usd=total,
                ))
                in_pair = False
                direction = 0
                accrued_funding_usd = 0.0
        else:
            new_direction = 0
            if bps >= p.entry_threshold_bps and avg_bps >= p.entry_avg_threshold_bps:
                new_direction = 1
            elif bps <= -p.entry_threshold_bps and avg_bps <= -p.entry_avg_threshold_bps \
                    and p.allow_negative_direction:
                new_direction = -1
            if new_direction != 0 and abs(basis_bps) <= p.basis_entry_block_bps:
                if new_direction == 1:
                    spot_entry = spot_px * (1 + slip_one_way)   # BUY spot
                    perp_entry = perp_px * (1 - slip_one_way)   # SELL perp
                else:
                    spot_entry = spot_px * (1 - slip_one_way)   # SELL spot (margin borrow)
                    perp_entry = perp_px * (1 + slip_one_way)   # BUY perp
                spot_qty = p.notional_per_pair_usd / spot_entry
                perp_qty = p.notional_per_pair_usd / perp_entry
                entry_spot_px = spot_entry
                entry_perp_px = perp_entry
                entry_ts_ms = t
                spot_fee = spot_qty * spot_entry * spot_fee_one_way
                perp_fee = perp_qty * perp_entry * perp_fee_one_way
                accrued_funding_usd = -(spot_fee + perp_fee)
                in_pair = True
                direction = new_direction

    # Close any dangling pair at last available reference (direction-aware)
    if in_pair:
        last_t = max(spot_by_t.keys()) if spot_by_t else 0
        spot_last = lookup_px(spot_by_t, last_t) or entry_spot_px
        perp_last = lookup_px(perp_by_t, last_t) or spot_last
        if direction == 1:
            spot_exit = spot_last * (1 - slip_one_way)
            perp_exit = perp_last * (1 + slip_one_way)
            spot_pnl = (spot_exit - entry_spot_px) * spot_qty
            perp_pnl = (entry_perp_px - perp_exit) * perp_qty
        else:
            spot_exit = spot_last * (1 + slip_one_way)
            perp_exit = perp_last * (1 - slip_one_way)
            spot_pnl = (entry_spot_px - spot_exit) * spot_qty
            perp_pnl = (perp_exit - entry_perp_px) * perp_qty
        spot_fee = spot_qty * spot_exit * spot_fee_one_way
        perp_fee = perp_qty * perp_exit * perp_fee_one_way
        total = accrued_funding_usd + spot_pnl + perp_pnl - spot_fee - perp_fee
        trades.append(SimTrade(
            symbol=symbol, strategy="funding_harvest",
            side=("neutral+" if direction == 1 else "neutral-"),
            qty=spot_qty, entry_price=entry_spot_px,
            stop=0.0, tp=0.0, entry_ts_ms=entry_ts_ms,
            exit_price=spot_exit, exit_reason="end_of_data",
            exit_ts_ms=last_t, pnl_usd=total,
        ))

    span_days = (funding_rows[-1]["fundingTime"] - funding_rows[0]["fundingTime"]) / 1000 / 86400
    return _stats_from_trades("funding_harvest", trades, s.account_equity_usd, span_days), trades


@dataclass
class FundingBacktestParams:
    """Mirror of live `HarvestParams` for backtest parity. Defaults must match
    live so a backtest with defaults predicts paper-mode behavior."""
    notional_per_pair_usd: float = 100.0
    entry_threshold_bps: float = 10.0
    entry_avg_threshold_bps: float = 5.0
    exit_threshold_bps: float = 2.0
    perp_adverse_move_pct: float = 30.0
    basis_entry_block_bps: float = 50.0
    basis_exit_alert_bps: float = 150.0
    allow_negative_direction: bool = True
    spot_borrow_bps_per_8h: float = 1.5


# ─────────────────────────────  Cascade-breakout backtest (v2)  ─────────────


@dataclass
class CascadeBacktestParams:
    """Execution-layer knobs for `simulate_cascade_breakout`. Defaults from
    research synthesis (see `project-cascade-strategy-research` memory)."""
    # Pattern-detector knobs flow through this so callers can tune everything
    # via one params object.
    detector: CascadeParams = field(default_factory=CascadeParams)

    # Stop placement
    stop_atr_mult: float = 1.5            # entry - 1.5 × ATR is the volatility cap
    stop_struct_buffer_atr: float = 0.25  # structural stop = struct_low - 0.25×ATR

    # Partial-fill take-profit
    tp1_r_multiple: float = 1.0           # +1.0R triggers 50% exit + stop→entry
    tp1_exit_fraction: float = 0.5

    # Trailing stop on remainder
    trail_atr_mult: float = 1.0           # chandelier: extreme − 1.0×ATR

    # Scratch — abort if no follow-through
    scratch_bars: int = 3                 # bars after entry to check follow-through
    scratch_retrace_pct: float = 0.50     # ≥50% retrace back into range = scratch

    # Hard time-stop
    hard_time_stop_bars: int = 24         # 24 M15 bars = 6h

    # Gate: minimum pattern confluence to take the trade
    min_confluence: int = 2

    # Risk + cost overrides (otherwise from Settings)
    risk_per_trade_pct: Optional[float] = None
    cost_bps_override: Optional[float] = None  # one-leg slip+fee in bps; round-trip = 2×


def simulate_cascade_breakout(
    symbol: str,
    ks: list[Kline],
    *,
    params: Optional[CascadeBacktestParams] = None,
    settings: Optional[Settings] = None,
    market: str = "perps",
) -> tuple[BacktestStats, list[SimTrade]]:
    """Iterate bar-by-bar through M15 klines, run the cascade pattern detector
    at each closed bar, and simulate the resulting trades with the v1
    execution rules.

    Execution model:
      Entry  : market at bar close after a pattern triggers
      Stop   : max(struct_low − 0.25×ATR, entry − 1.5×ATR)
      TP1    : +1R; exit 50% and move stop to entry
      Trail  : 1.0×ATR chandelier on remainder
      Scratch: ≥50% retrace into range within `scratch_bars` of entry
      Hard   : 24 M15 bars (6h) time-stop

    Stop fires BEFORE TP within a single bar (conservative for backtest)."""
    s = settings or get_settings()
    p = params or CascadeBacktestParams()

    warmup_n = max(p.detector.cascade_lookback_bars + 30, 100)
    if len(ks) < warmup_n + 10:
        return _stats_from_trades("cascade_breakout", [], s.account_equity_usd, 0.0), []

    risk_pct = (p.risk_per_trade_pct or s.risk_per_trade_pct) / 100.0
    taker_bps = s.perps_taker_fee_bps if market == "perps" else s.spot_taker_fee_bps
    if p.cost_bps_override is not None:
        fee_bps = p.cost_bps_override
    else:
        fee_bps = taker_bps + s.slippage_bps
    stop_slip = s.paper_stop_slippage_bps / 10_000
    tp_slip = s.paper_tp_slippage_bps / 10_000

    equity = s.account_equity_usd

    trades: list[SimTrade] = []
    open_trade: Optional[SimTrade] = None
    cooldown_until_ms: int = 0

    for i in range(warmup_n, len(ks)):
        k = ks[i]

        # ─── Resolve any open trade against THIS bar first ───
        if open_trade is not None:
            open_trade.bars_held += 1  # type: ignore[attr-defined]
            side = open_trade.side
            phase = open_trade.phase  # type: ignore[attr-defined]

            stop_hit = ((side == "long" and k.low <= open_trade.stop)
                        or (side == "short" and k.high >= open_trade.stop))

            if phase == "pre_tp1":
                tp1 = open_trade.tp1_price  # type: ignore[attr-defined]
                tp1_hit = ((side == "long" and k.high >= tp1)
                           or (side == "short" and k.low <= tp1))

                if stop_hit:
                    _close_cascade_trade(open_trade, open_trade.stop, "stop",
                                         k.close_time, stop_slip, fee_bps, full=True)
                    trades.append(open_trade)
                    cooldown_until_ms = k.close_time + 60 * 60_000
                    open_trade = None
                elif tp1_hit:
                    _take_partial(open_trade, tp1, k.close_time, tp_slip,
                                  fee_bps, p.tp1_exit_fraction)
                    open_trade.phase = "post_tp1"        # type: ignore[attr-defined]
                    open_trade.stop = open_trade.entry_price
                    open_trade.trail_extreme = (         # type: ignore[attr-defined]
                        k.high if side == "long" else k.low)
                elif (open_trade.bars_held >= p.scratch_bars                  # type: ignore[attr-defined]
                      and _scratch_check(open_trade, k, p)):
                    _close_cascade_trade(open_trade, k.close, "scratch",
                                         k.close_time, 0.0, fee_bps, full=True)
                    trades.append(open_trade)
                    open_trade = None
                elif open_trade.bars_held >= p.hard_time_stop_bars:           # type: ignore[attr-defined]
                    _close_cascade_trade(open_trade, k.close, "time_stop",
                                         k.close_time, 0.0, fee_bps, full=True)
                    trades.append(open_trade)
                    open_trade = None

            elif phase == "post_tp1":
                # Update trail extreme then check stop
                if side == "long":
                    open_trade.trail_extreme = max(                           # type: ignore[attr-defined]
                        open_trade.trail_extreme, k.high)                    # type: ignore[attr-defined]
                    trail_stop = open_trade.trail_extreme - (                # type: ignore[attr-defined]
                        p.trail_atr_mult * open_trade.atr_at_entry)          # type: ignore[attr-defined]
                    open_trade.stop = max(open_trade.stop, trail_stop)
                else:
                    open_trade.trail_extreme = min(                           # type: ignore[attr-defined]
                        open_trade.trail_extreme, k.low)                     # type: ignore[attr-defined]
                    trail_stop = open_trade.trail_extreme + (                # type: ignore[attr-defined]
                        p.trail_atr_mult * open_trade.atr_at_entry)          # type: ignore[attr-defined]
                    open_trade.stop = min(open_trade.stop, trail_stop)

                stop_hit_post = ((side == "long" and k.low <= open_trade.stop)
                                  or (side == "short" and k.high >= open_trade.stop))
                if stop_hit_post:
                    _close_cascade_trade(open_trade, open_trade.stop, "trail_stop",
                                         k.close_time, stop_slip, fee_bps, full=False)
                    trades.append(open_trade)
                    open_trade = None
                elif open_trade.bars_held >= p.hard_time_stop_bars:           # type: ignore[attr-defined]
                    _close_cascade_trade(open_trade, k.close, "time_stop",
                                         k.close_time, 0.0, fee_bps, full=False)
                    trades.append(open_trade)
                    open_trade = None

        if open_trade is not None or k.close_time < cooldown_until_ms:
            continue

        # ─── Detect pattern at this bar ───
        slice_ks = ks[: i + 1]
        pat = detect_pattern(slice_ks, params=p.detector)
        if pat is None or pat.confluence_count < p.min_confluence:
            continue

        atr = _atr(slice_ks, period=14)
        if atr is None or atr <= 0:
            continue

        # Stop placement: max of structural buffer and volatility cap
        if pat.side == "long":
            chain_lows = [pp.price for pp in pat.cascade.pivots if pp.kind == "low"]
            struct = min(chain_lows) if chain_lows else (k.close - atr)
            stop = max(struct - p.stop_struct_buffer_atr * atr,
                       k.close - p.stop_atr_mult * atr)
        else:
            chain_highs = [pp.price for pp in pat.cascade.pivots if pp.kind == "high"]
            struct = max(chain_highs) if chain_highs else (k.close + atr)
            stop = min(struct + p.stop_struct_buffer_atr * atr,
                       k.close + p.stop_atr_mult * atr)

        risk_per_unit = abs(k.close - stop)
        if risk_per_unit <= 0:
            continue

        entry_slip = k.close * (s.slippage_bps / 10_000)
        entry_px = k.close + entry_slip if pat.side == "long" else k.close - entry_slip
        if pat.side == "long":
            tp1 = entry_px + p.tp1_r_multiple * risk_per_unit
        else:
            tp1 = entry_px - p.tp1_r_multiple * risk_per_unit

        risk_usd = equity * risk_pct
        qty = min(risk_usd / risk_per_unit, s.max_notional_usd / entry_px)
        if qty <= 0:
            continue

        open_trade = SimTrade(
            symbol=symbol, strategy=f"cascade_{pat.mode}",
            side=pat.side, qty=qty,
            entry_price=entry_px, stop=stop, tp=tp1,
            entry_ts_ms=k.close_time,
        )
        # Attach dynamic state for the simulator loop
        open_trade.phase = "pre_tp1"             # type: ignore[attr-defined]
        open_trade.tp1_price = tp1               # type: ignore[attr-defined]
        open_trade.partial_pnl = 0.0             # type: ignore[attr-defined]
        open_trade.partial_qty = 0.0             # type: ignore[attr-defined]
        open_trade.bars_held = 0                 # type: ignore[attr-defined]
        open_trade.atr_at_entry = atr            # type: ignore[attr-defined]
        open_trade.trail_extreme = None          # type: ignore[attr-defined]
        open_trade.entry_high = k.high           # type: ignore[attr-defined]
        open_trade.entry_low = k.low             # type: ignore[attr-defined]

    if open_trade is not None and ks:
        last = ks[-1].close
        _close_cascade_trade(open_trade, last, "eod", ks[-1].close_time,
                             0.0, fee_bps,
                             full=(open_trade.phase == "pre_tp1"))  # type: ignore[attr-defined]
        trades.append(open_trade)

    span_days = (ks[-1].close_time - ks[0].close_time) / 1000 / 86400 if ks else 0.0
    return _stats_from_trades("cascade_breakout", trades,
                               s.account_equity_usd, span_days), trades


def _take_partial(t: SimTrade, tp_price: float, ts_ms: int,
                   tp_slip: float, fee_bps: float, frac: float) -> None:
    """Realize partial PnL on `frac` of the trade qty at tp_price."""
    fill_px = tp_price * (1 - tp_slip) if t.side == "long" else tp_price * (1 + tp_slip)
    partial_qty = t.qty * frac
    gross = (fill_px - t.entry_price) * partial_qty
    if t.side == "short":
        gross = -gross
    fee = partial_qty * fill_px * (fee_bps / 10_000)
    t.partial_pnl = (t.partial_pnl or 0.0) + (gross - fee)  # type: ignore[attr-defined]
    t.partial_qty = (t.partial_qty or 0.0) + partial_qty    # type: ignore[attr-defined]


def _close_cascade_trade(t: SimTrade, exit_price: float, reason: str,
                          ts_ms: int, slip: float, fee_bps: float, full: bool) -> None:
    """Close the remaining qty. If full=False, only the non-partial-filled
    qty is closed here; partial PnL is added to total."""
    fill_px = exit_price * (1 - slip) if t.side == "long" else exit_price * (1 + slip)
    if full:
        remaining_qty = t.qty
    else:
        remaining_qty = t.qty - (t.partial_qty or 0.0)  # type: ignore[attr-defined]
    gross = (fill_px - t.entry_price) * remaining_qty
    if t.side == "short":
        gross = -gross
    fee = remaining_qty * fill_px * (fee_bps / 10_000)
    final_leg_pnl = gross - fee
    total = (t.partial_pnl or 0.0) + final_leg_pnl  # type: ignore[attr-defined]
    t.exit_price = fill_px
    t.exit_reason = reason
    t.exit_ts_ms = ts_ms
    t.pnl_usd = total


def _scratch_check(t: SimTrade, k: Kline, p: CascadeBacktestParams) -> bool:
    """Scratch if price has retraced ≥ scratch_retrace_pct into the range
    formed by the breakout bar. For a long: if low has dropped ≥ X% of
    (entry_high - entry_low) below the entry. Mirror for short."""
    if not hasattr(t, "entry_high") or not hasattr(t, "entry_low"):
        return False
    rng = t.entry_high - t.entry_low                  # type: ignore[attr-defined]
    if rng <= 0:
        return False
    if t.side == "long":
        retrace = (t.entry_price - k.low) / rng
    else:
        retrace = (k.high - t.entry_price) / rng
    return retrace >= p.scratch_retrace_pct


def format_stats(stats: BacktestStats) -> str:
    return (
        f"\n=== {stats.strategy} ===\n"
        f"trades:           {stats.trades}\n"
        f"win rate:         {stats.win_rate:.1%}\n"
        f"total P&L:        ${stats.total_pnl_usd:+.2f}\n"
        f"avg P&L / trade:  ${stats.avg_pnl_usd:+.2f}\n"
        f"Sharpe:           {stats.sharpe:.2f}\n"
        f"deflated Sharpe:  {stats.deflated_sharpe:.2f}\n"
        f"max drawdown:     ${stats.max_drawdown_usd:.2f} ({stats.max_drawdown_pct:.1f}%)\n"
        f"annualized:       {stats.annualized_pct:+.1f}%\n"
        f"start equity:     ${stats.starting_equity_usd:.2f}\n"
        f"end equity:       ${stats.ending_equity_usd:.2f}\n"
    )
