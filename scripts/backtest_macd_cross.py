"""Card — MACD histogram zero-cross reversal (decisive single test).

Faithful translation of the bundled "BitMEX simple trading robot" (Krotov,
Habr/Medium 2018). The bot's entire alpha is one rule in `strategy.py`:

    macd, signal, hist = talib.MACD(close, fastperiod=8, slowperiod=28, signalperiod=9)
    if hist[-2] > 0 and hist[-1] < 0:  return -1   # go short
    if hist[-2] < 0 and hist[-1] > 0:  return +1   # go long

i.e. **enter the direction of the MACD-histogram zero-cross on closed 1h XBTUSD
bars** and (in the AWS "working" variant) flip on the opposite cross. No stop,
no edge filter, single instrument. We model the canonical always-in-market
reversal: long after a bullish cross, short after a bearish cross, flip on the
opposite signal. Position notional = equity * leverage, costs ON (perp taker
fee + half-spread + Almgren-Chriss impact) charged on BOTH legs of every flip.

Economic prior (falsification-first): a fixed-parameter MACD cross on one
liquid major is the single most over-published technical signal in crypto. It
is a slow trend filter that whipsaws in range regimes; the question is purely
whether the trend capture survives crossing the spread + fee on every flip.
On 1h BTC the cross fires often, so fees are THE gate.

Stop rule (mirrors Card 1 Δfunding): if price-PnL net of fees on the headline
run (BTCUSDT 1h, costs ON) is <= 0, the idea is DEAD and we do NOT proceed to
a CPCV(10,2)+PBO grid. Funding accrual is NOT modeled — a flip strategy is
roughly funding-neutral over a long horizon, and the decisive gate is whether
directional price capture even clears commissions.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.backtest_macd_cross \\
      --symbol BTCUSDT --tf 1h --bars 8760 --leverage 1
  # robustness grid:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.backtest_macd_cross --grid
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from src.config.settings import get_settings
from src.models.types import Kline
from src.services.backtest import (
    BacktestStats,
    SimTrade,
    _adv_5m_usd,
    _stats_from_trades,
)
from src.services.costs import (
    Costs,
    adjust_entry_price,
    adjust_exit_price,
    impact_k_for_symbol,
    taker_fee_usd,
)
from src.tools.binance_client import BinanceClient


def _ema(prev: Optional[float], value: float, period: int) -> float:
    """Same EMA recurrence as src/tools/indicators._ema (causal, closed-bar)."""
    k = 2.0 / (period + 1)
    return value if prev is None else (value - prev) * k + prev


@dataclass
class MacdParams:
    fast: int = 8        # bot default
    slow: int = 28       # bot default
    signal: int = 9      # bot default
    leverage: float = 1.0


def _macd_hist_series(closes: list[float], p: MacdParams) -> list[Optional[float]]:
    """Causal MACD histogram, one value per bar (None until warmed up).

    hist = (EMA_fast - EMA_slow) - EMA_signal(EMA_fast - EMA_slow). Every value
    at index i uses only closes[0..i], so reading hist[i] at bar i's close is
    look-ahead free."""
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ema_sig: Optional[float] = None
    out: list[Optional[float]] = []
    for c in closes:
        ema_fast = _ema(ema_fast, c, p.fast)
        ema_slow = _ema(ema_slow, c, p.slow)
        macd_line = ema_fast - ema_slow
        ema_sig = _ema(ema_sig, macd_line, p.signal)
        out.append(macd_line - ema_sig)
    return out


def simulate_macd_cross(
    ks: list[Kline],
    symbol: str,
    p: MacdParams,
    start_equity: float,
    costs: Costs,
    venue: str = "perp",
) -> tuple[BacktestStats, list[SimTrade]]:
    """Always-in-market MACD-histogram zero-cross reversal.

    Long after hist crosses up through 0; short after it crosses down. Flip on
    the opposite cross, closing the old leg and opening the new one at the SAME
    bar close. Both legs pay taker fee + adverse slippage. One SimTrade per leg
    (entry -> the flip that closes it). Notional = start_equity * leverage,
    fixed (non-compounding) so PnL is comparable across runs."""
    closes = [k.close for k in ks]
    hist = _macd_hist_series(closes, p)
    impact_k = impact_k_for_symbol(symbol)
    half_spread = costs.half_spread_bps_default
    notional = start_equity * p.leverage

    trades: list[SimTrade] = []
    open_trade: Optional[SimTrade] = None

    # Warmup: require slow+signal bars before trusting the histogram.
    warm = p.slow + p.signal + 1
    for i in range(warm, len(ks)):
        h_prev, h_now = hist[i - 1], hist[i]
        if h_prev is None or h_now is None:
            continue
        signal_side: Optional[str] = None
        if h_prev < 0 and h_now > 0:
            signal_side = "long"
        elif h_prev > 0 and h_now < 0:
            signal_side = "short"
        if signal_side is None:
            continue
        if open_trade is not None and open_trade.side == signal_side:
            continue  # already positioned this way; no churn

        k = ks[i]
        adv5m = _adv_5m_usd(ks, i)
        raw_px = k.close

        # Close the existing leg at this bar's close (adverse exit slippage + fee).
        if open_trade is not None:
            exit_px = adjust_exit_price(
                raw_px, open_trade.side, abs(open_trade.qty * raw_px),
                adv5m, impact_k, half_spread,
            )
            gross = (exit_px - open_trade.entry_price) * open_trade.qty
            if open_trade.side == "short":
                gross = -gross
            exit_fee = taker_fee_usd(abs(open_trade.qty * exit_px), venue, costs)
            open_trade.exit_price = exit_px
            open_trade.exit_reason = "flip"
            open_trade.exit_ts_ms = k.close_time
            # entry fee was already netted into pnl at open; subtract exit fee now.
            open_trade.pnl_usd = (open_trade.pnl_usd or 0.0) + gross - exit_fee
            trades.append(open_trade)
            open_trade = None

        # Open the new leg at this bar's close (adverse entry slippage + fee).
        entry_px = adjust_entry_price(
            raw_px, signal_side, notional, adv5m, impact_k, half_spread,
        )
        qty = notional / entry_px
        entry_fee = taker_fee_usd(notional, venue, costs)
        open_trade = SimTrade(
            symbol=symbol, strategy="macd_cross", side=signal_side, qty=qty,
            entry_price=entry_px, stop=0.0, tp=0.0, entry_ts_ms=k.close_time,
            pnl_usd=-entry_fee,  # carry entry fee until the leg closes
        )

    # Close the dangling leg at the last bar's close.
    if open_trade is not None and ks:
        last = ks[-1]
        adv5m = _adv_5m_usd(ks, len(ks) - 1)
        exit_px = adjust_exit_price(
            last.close, open_trade.side, abs(open_trade.qty * last.close),
            adv5m, impact_k, half_spread,
        )
        gross = (exit_px - open_trade.entry_price) * open_trade.qty
        if open_trade.side == "short":
            gross = -gross
        exit_fee = taker_fee_usd(abs(open_trade.qty * exit_px), venue, costs)
        open_trade.exit_price = exit_px
        open_trade.exit_reason = "eod"
        open_trade.exit_ts_ms = last.close_time
        open_trade.pnl_usd = (open_trade.pnl_usd or 0.0) + gross - exit_fee
        trades.append(open_trade)

    span_days = (ks[-1].close_time - ks[0].close_time) / 1000 / 86400 if ks else 0.0
    stats = _stats_from_trades("macd_cross", trades, start_equity, span_days)
    return stats, trades


def _gross_price_pnl(trades: list[SimTrade], costs: Costs, venue: str) -> float:
    """Reconstruct price-PnL net of fees but BEFORE... no — we charged fees into
    pnl. Here we recover total fees so we can report price-net-of-fees vs total.
    Each leg pays 2 taker fees (entry+exit) on ~equal notional."""
    fees = 0.0
    for t in trades:
        if t.exit_price is None:
            continue
        fees += taker_fee_usd(abs(t.qty * t.entry_price), venue, costs)
        fees += taker_fee_usd(abs(t.qty * t.exit_price), venue, costs)
    return fees


async def _fetch_klines(b: BinanceClient, symbol: str, tf: str, bars: int,
                        market: str) -> list[Kline]:
    raw = await b.fetch_klines_paginated(symbol, tf, total=bars, market=market)
    return [Kline(
        symbol=symbol, timeframe=tf,
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    ) for r in raw]


def _print_run(label: str, stats: BacktestStats, trades: list[SimTrade],
               costs: Costs, venue: str, equity: float, headline: bool) -> bool:
    fees = _gross_price_pnl(trades, costs, venue)
    total = stats.total_pnl_usd
    price_net_of_fees = total  # fees already netted into pnl
    price_gross = total + fees
    print("=" * 90)
    print(f"  {label}")
    print("-" * 90)
    print(f"  trades (flips):   {stats.trades}")
    print(f"  total PnL (net):  ${total:+.2f}  ({stats.annualized_pct:+.1f}%/yr on ${equity:.0f})")
    print(f"  PnL ex-fees:      ${price_gross:+.2f}   (slippage still removed)")
    print(f"  total fees paid:  ${fees:+.2f}")
    print(f"  net of ALL costs: ${price_net_of_fees:+.2f}   <-- stop rule")
    print(f"  win rate:         {stats.win_rate*100:.0f}%  ({stats.wins}/{stats.trades})")
    print(f"  Sharpe:           {stats.sharpe:+.2f}")
    print(f"  Deflated Sharpe:  {stats.deflated_sharpe:+.2f}")
    print(f"  max drawdown:     {stats.max_drawdown_pct:.1f}%")
    alive = price_net_of_fees > 0
    if headline:
        verdict = ("ALIVE — clears the stop rule; worth the full CPCV(10,2)+PBO grid."
                   if alive else
                   "DEAD — price net of fees <= 0; do NOT run the CPCV grid.")
        print(f"  VERDICT: {verdict}")
    return alive


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="1h", help="bot default is 1h")
    ap.add_argument("--bars", type=int, default=8760, help="~365d of 1h bars")
    ap.add_argument("--fast", type=int, default=8)
    ap.add_argument("--slow", type=int, default=28)
    ap.add_argument("--signal", type=int, default=9)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--equity-usd", type=float, default=1_000.0)
    ap.add_argument("--market", default="perps", choices=["spot", "perps"])
    ap.add_argument("--grid", action="store_true",
                    help="run robustness grid BTC/ETH/SOL x 1h/4h instead of single run")
    args = ap.parse_args()

    s = get_settings()
    equity = args.equity_usd or s.account_equity_usd
    costs = Costs()
    venue = "perp" if args.market == "perps" else "spot"

    b = BinanceClient()
    await b.start()
    try:
        if not args.grid:
            p = MacdParams(args.fast, args.slow, args.signal, args.leverage)
            ks = await _fetch_klines(b, args.symbol, args.tf, args.bars, args.market)
            span = (ks[-1].close_time - ks[0].close_time) / 1000 / 86400 if ks else 0
            stats, trades = simulate_macd_cross(ks, args.symbol, p, equity, costs, venue)
            print()
            _print_run(
                f"MACD({p.fast},{p.slow},{p.signal}) zero-cross  ·  {args.symbol} "
                f"{args.tf}  ·  {len(ks)} bars (~{span:.0f}d)  ·  {args.leverage:g}x  ·  COSTS ON",
                stats, trades, costs, venue, equity, headline=True,
            )
            print("=" * 90)
            return

        # Robustness grid.
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        tfs = [("1h", 8760), ("4h", 4380)]
        print("\nMACD ZERO-CROSS ROBUSTNESS GRID  ·  costs ON  ·  1x  ·  "
              "headline = BTCUSDT 1h\n")
        alive_count = 0
        total_runs = 0
        for sym in symbols:
            for tf, bars in tfs:
                p = MacdParams(args.fast, args.slow, args.signal, 1.0)
                ks = await _fetch_klines(b, sym, tf, bars, args.market)
                span = (ks[-1].close_time - ks[0].close_time) / 1000 / 86400 if ks else 0
                stats, trades = simulate_macd_cross(ks, sym, p, equity, costs, venue)
                headline = (sym == "BTCUSDT" and tf == "1h")
                alive = _print_run(
                    f"{sym} {tf}  ·  {len(ks)} bars (~{span:.0f}d)"
                    + ("   [HEADLINE]" if headline else ""),
                    stats, trades, costs, venue, equity, headline=headline,
                )
                alive_count += int(alive)
                total_runs += 1
        print("=" * 90)
        print(f"  GRID SUMMARY: {alive_count}/{total_runs} runs price-net-of-fees positive")
        print("=" * 90)
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
