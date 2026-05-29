"""Card 1 — Δfunding cross-sectional spread (decisive single test).

Identical machinery to the validated cross-sectional carry backtest
(`scripts/backtest_funding_carry.py`), but the per-symbol ranking signal is the
*change* in funding rather than its level:

    signal_i(ts) = mean(funding_i over [ts-w, ts)) − mean(funding_i over [ts-2w, ts-w))

Longs = fastest-RISING funding, shorts = fastest-FALLING, dollar-neutral.
Thesis (researcher Card 1): funding levels are ~0.97-0.99 autocorrelated, so the
level is near a unit root carrying little new information and is exactly what
Ethena/BFUSD structural-short inventory compressed in 2025; the first difference
(repricing surprise) is near-orthogonal and harder to arbitrage.

This is the **cheapest decisive test** from the hypothesis memo: one 365d,
top-30, costs-ON run. Stop rule (per the memo): if total PnL net of costs — and
in particular price-PnL minus costs — is negative, the idea is dead and we do
NOT proceed to the full CPCV grid.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.backtest_funding_delta \\
      --days 365 --top-n-universe 30 --top-n 3 --window-hours 168
"""
from __future__ import annotations

import argparse
import asyncio
import math
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from scripts.backtest_funding_carry import (
    REPO,
    SymbolHistory,
    build_universe,
    fetch_funding_history,
    fetch_perp_closes_8h,
    simulate_carry,
    summarise,
)
from src.scanners.universe_pit import SymbolListing, load_pit_log
from src.services.costs import Costs
from src.strategies.funding_carry import CarryParams, funding_window_change
from src.tools.binance_client import BinanceClient


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--top-n", type=int, default=3,
                    help="positions per leg (longs + shorts)")
    ap.add_argument("--top-n-universe", type=int, default=30,
                    help="universe size (top-N USDT perps by 24h volume)")
    ap.add_argument("--book-pct-per-side", type=float, default=0.25)
    ap.add_argument("--rebalance-hours", type=int, default=168)
    ap.add_argument("--window-hours", type=int, default=168,
                    help="Δfunding window w (trailing minus prior, each w long)")
    ap.add_argument("--equity-usd", type=float, default=1_000.0)
    ap.add_argument("--pit-log",
                    default="data/research/universe/binance_delistings.json",
                    help="PIT listings JSON. Set to '' to disable PIT correction.")
    args = ap.parse_args()

    pit_log: Optional[dict[str, SymbolListing]] = None
    if args.pit_log:
        pit_path = Path(args.pit_log)
        if not pit_path.is_absolute():
            pit_path = REPO / pit_path
        pit_log = load_pit_log(pit_path)
        if not pit_log:
            print(f"WARNING: --pit-log={pit_path} not found; running WITHOUT "
                  "survivorship correction.")
            pit_log = None
        else:
            print(f"PIT log: {len(pit_log)} symbols loaded from {pit_path}")
    else:
        print("PIT correction DISABLED (--pit-log='')")

    p = CarryParams(top_n=args.top_n,
                    rebalance_period_hours=args.rebalance_hours,
                    book_pct_per_side=args.book_pct_per_side)
    costs = Costs()
    window_hours = args.window_hours

    # Δfunding ranking signal — closes over window_hours; signature matches the
    # default `_funding_rate_at(history, ts)` that simulate_carry expects.
    def delta_signal(funding_events, ts_ms):
        return funding_window_change(funding_events, ts_ms, window_hours=window_hours)

    b = BinanceClient()
    await b.start()
    try:
        universe = await build_universe(b, top_n_universe=args.top_n_universe)
        print(f"universe ({len(universe)} symbols):\n  {', '.join(universe[:10])}"
              f"{'...' if len(universe) > 10 else ''}")

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - args.days * 86_400_000
        # Δfunding at the first rebalance (ts=start_ms) needs the prior 2 windows
        # of funding history, so fetch funding starting 2w earlier.
        warmup_ms = 2 * window_hours * 3_600_000
        funding_start_ms = start_ms - warmup_ms

        histories: dict[str, SymbolHistory] = {}
        bars_8h = math.ceil(args.days * 3) + math.ceil(warmup_ms / (8 * 3_600_000)) + 10
        t0 = time.time()
        for i, sym in enumerate(universe, 1):
            funding, closes = await asyncio.gather(
                fetch_funding_history(b, sym, funding_start_ms, now_ms),
                fetch_perp_closes_8h(b, sym, bars_8h),
            )
            histories[sym] = SymbolHistory(funding=funding, closes_8h=closes)
            if i % 5 == 0 or i == len(universe):
                print(f"   [{i}/{len(universe)}] {sym}  funding={len(funding)}  "
                      f"closes={len(closes)}  ({time.time()-t0:.0f}s)", flush=True)

        results = simulate_carry(
            histories, start_ms=start_ms, end_ms=now_ms,
            p=p, start_equity=args.equity_usd, costs=costs,
            pit_log=pit_log, signal_fn=delta_signal,
        )
        span_days = (now_ms - start_ms) / 86_400_000
        stats = summarise(results, start_equity=args.equity_usd, span_days=span_days)

        long_px = stats.get("long_price_pnl", 0.0)
        short_px = stats.get("short_price_pnl", 0.0)
        fee = stats.get("fee_pnl", 0.0)
        long_fnd = stats.get("long_funding_pnl", 0.0)
        short_fnd = stats.get("short_funding_pnl", 0.0)
        price_pnl = long_px + short_px
        funding_pnl = long_fnd + short_fnd

        print("\n" + "=" * 90)
        print(f"Δ-FUNDING (Card 1) BACKTEST  ·  {args.days}d  ·  top {p.top_n} L/S "
              f"·  {p.book_pct_per_side*100:.0f}%/side  ·  w={window_hours}h  ·  COSTS ON")
        print("=" * 90)
        print(f"  weeks:            {stats['weeks']}")
        print(f"  total PnL:        ${stats['total_pnl_usd']:+.2f}  "
              f"({stats['annualized_pct']:+.1f}%/yr on ${args.equity_usd:.0f})")
        print(f"  price PnL:        ${price_pnl:+.2f}   (long {long_px:+.2f} / short {short_px:+.2f})")
        print(f"  funding PnL:      ${funding_pnl:+.2f}   (long {long_fnd:+.2f} / short {short_fnd:+.2f})")
        print(f"  fee PnL:          ${fee:+.2f}")
        print(f"  price − fees:     ${price_pnl + fee:+.2f}   <-- memo stop rule")
        print(f"  win rate:         {stats['win_rate']*100:.0f}%  ({stats['win_weeks']}/{stats['weeks']})")
        print(f"  Sharpe:           {stats['sharpe']:+.2f}")
        print(f"  Deflated Sharpe:  {stats['deflated_sharpe']:+.2f}")
        print(f"  max drawdown:     {stats['max_drawdown_pct']:.1f}%")
        print("=" * 90)
        verdict = ("DEAD — price net of fees is negative; do NOT run the CPCV grid."
                   if (price_pnl + fee) <= 0
                   else "ALIVE — clears the stop rule; worth the full CPCV(10,2)+PBO grid.")
        print(f"  VERDICT: {verdict}")
        print("=" * 90)
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
