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

    raw = await binance.fetch_klines(symbol, tf, limit=bars, market="spot")
    ks = [Kline(
        symbol=symbol, timeframe=tf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in raw]
    htf_raw = await binance.fetch_klines(symbol, htf, limit=max(300, bars // 12),
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

    for k in ks[warmup_n:]:
        # Resolve any open trade against THIS bar first (price might have hit stop/TP).
        if open_trade is not None:
            hit_stop = (k.low <= open_trade.stop) if open_trade.side == "long" else (k.high >= open_trade.stop)
            hit_tp = (k.high >= open_trade.tp) if open_trade.side == "long" else (k.low <= open_trade.tp)
            if hit_stop:
                # Adverse slippage on stop (25 bps)
                exit_px = open_trade.stop * (1 - 25 / 10_000) if open_trade.side == "long" else open_trade.stop * (1 + 25 / 10_000)
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
                exit_px = open_trade.tp * (1 - 5 / 10_000) if open_trade.side == "long" else open_trade.tp * (1 + 5 / 10_000)
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
