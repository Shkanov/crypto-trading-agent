"""Card 2 — carry conditioned on momentum agreement (decay-defense test).

The funding-edges research memo argues plain cross-sectional carry is *decaying*
(Cao SSRN 6365329 + BitMEX 2025: Ethena/BFUSD structural-short inventory
compressed the yield to sub-T-bill in 2025). Card 2 is a *defense*, not a new
edge: run the validated carry overlay but keep only the high-funding longs /
low-funding shorts whose trailing momentum AGREES with the carry direction —
dropping exactly the pure-yield names that crowding hollowed out.

The relevant null is NOT zero — it is **plain carry**. The overlay only earns a
slot if it beats plain carry net of costs. This driver therefore runs both on
the SAME fetched universe + histories + PIT log, so the comparison is exact, and
applies the memo's decisive cheap-test discipline before any CPCV grid.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.backtest_funding_carry_momentum \\
      --days 365 --top-n-universe 30 --top-n 3 --momentum-days 14
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
    WeeklyResult,
    _funding_rate_at,
    _nearest_price,
    build_universe,
    fetch_funding_history,
    fetch_perp_closes_8h,
    simulate_carry,
    summarise,
)
from src.scanners.universe_pit import SymbolListing, is_active_at, load_pit_log
from src.services.costs import Costs
from src.strategies.funding_carry import (
    CarryParams,
    CarryPosition,
    cycle_pnl,
    price_momentum,
    rank_for_carry_momentum,
)
from src.tools.binance_client import BinanceClient


def simulate_carry_momentum(
    histories: dict[str, SymbolHistory],
    start_ms: int,
    end_ms: int,
    p: CarryParams,
    start_equity: float,
    costs: Costs,
    momentum_lookback_hours: int,
    pit_log: Optional[dict[str, SymbolListing]] = None,
) -> list[WeeklyResult]:
    """Same weekly walk as `simulate_carry`, but the rebalance keeps only
    funding-ranked names whose momentum agrees (`rank_for_carry_momentum`).
    Per-leg notional uses a fixed `top_n` denominator, so a momentum-dropped
    slot leaves that capital undeployed (faithful to the breadth-loss risk)."""
    rebalance_step = p.rebalance_period_hours * 3_600_000
    results: list[WeeklyResult] = []
    equity = start_equity
    ts = start_ms

    while ts + rebalance_step <= end_ms:
        next_ts = ts + rebalance_step

        funding_snap: dict[str, float] = {}
        mom_snap: dict[str, float] = {}
        pit_drops = 0
        for sym, hist in histories.items():
            r = _funding_rate_at(hist.funding, ts)
            if r is None:
                continue
            if pit_log is not None and not is_active_at(pit_log, sym, ts):
                pit_drops += 1
                continue
            funding_snap[sym] = r
            m = price_momentum(hist.closes_8h, ts, momentum_lookback_hours)
            if m is not None:
                mom_snap[sym] = m

        # Breadth guard mirrors build_rebalance.
        n_universe = len(funding_snap)
        if n_universe < p.min_universe_size or n_universe < 2 * p.top_n:
            ts = next_ts
            continue

        longs, shorts = rank_for_carry_momentum(funding_snap, mom_snap, p)
        if not longs and not shorts:
            ts = next_ts
            continue

        # Fixed top_n denominator → masked slots stay undeployed.
        per_position = equity * p.book_pct_per_side / p.top_n
        positions = (
            [CarryPosition(symbol=s, side="long", notional_usd=per_position,
                           entry_funding_rate=funding_snap[s]) for s in longs]
            + [CarryPosition(symbol=s, side="short", notional_usd=per_position,
                             entry_funding_rate=funding_snap[s]) for s in shorts]
        )

        weekly_pnl = 0.0
        long_px = short_px = long_fnd = short_fnd = fee = 0.0
        for pos in positions:
            hist = histories[pos.symbol]
            entry_px = _nearest_price(hist.closes_8h, ts)
            exit_px = _nearest_price(hist.closes_8h, next_ts)
            if entry_px is None or exit_px is None:
                continue
            res = cycle_pnl(pos, entry_price=entry_px, exit_price=exit_px,
                            funding_events=hist.funding,
                            entry_ts_ms=ts, exit_ts_ms=next_ts, costs=costs)
            weekly_pnl += res.total_pnl_usd
            fee += res.fee_pnl_usd
            if pos.side == "long":
                long_px += res.price_pnl_usd
                long_fnd += res.funding_pnl_usd
            else:
                short_px += res.price_pnl_usd
                short_fnd += res.funding_pnl_usd

        equity += weekly_pnl
        results.append(WeeklyResult(
            rebalance_ts_ms=ts, universe_n=n_universe,
            longs=[s for s in longs], shorts=[s for s in shorts],
            long_price_pnl=long_px, short_price_pnl=short_px,
            long_funding_pnl=long_fnd, short_funding_pnl=short_fnd,
            fee_pnl=fee, total_pnl_usd=weekly_pnl, pit_filtered=pit_drops,
        ))
        ts = next_ts

    return results


def _line(tag: str, stats: dict) -> str:
    price = stats.get("long_price_pnl", 0.0) + stats.get("short_price_pnl", 0.0)
    return (f"  {tag:14s} PnL=${stats['total_pnl_usd']:+8.2f}  "
            f"price-fees=${price + stats.get('fee_pnl', 0.0):+8.2f}  "
            f"Sharpe={stats['sharpe']:+.2f}  DSR={stats['deflated_sharpe']:+.2f}  "
            f"win={stats['win_rate']*100:3.0f}%  maxDD={stats['max_drawdown_pct']:.1f}%")


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--top-n-universe", type=int, default=30)
    ap.add_argument("--book-pct-per-side", type=float, default=0.25)
    ap.add_argument("--rebalance-hours", type=int, default=168)
    ap.add_argument("--momentum-days", type=int, default=14,
                    help="trailing price-momentum lookback for the agreement mask")
    ap.add_argument("--equity-usd", type=float, default=1_000.0)
    ap.add_argument("--pit-log",
                    default="data/research/universe/binance_delistings.json")
    args = ap.parse_args()

    pit_log: Optional[dict[str, SymbolListing]] = None
    if args.pit_log:
        pit_path = Path(args.pit_log)
        if not pit_path.is_absolute():
            pit_path = REPO / pit_path
        pit_log = load_pit_log(pit_path) or None
        print(f"PIT log: {len(pit_log)} symbols" if pit_log
              else "PIT correction DISABLED")

    p = CarryParams(top_n=args.top_n, rebalance_period_hours=args.rebalance_hours,
                    book_pct_per_side=args.book_pct_per_side)
    costs = Costs()
    mom_hours = args.momentum_days * 24

    b = BinanceClient()
    await b.start()
    try:
        universe = await build_universe(b, top_n_universe=args.top_n_universe)
        print(f"universe ({len(universe)}): {', '.join(universe[:10])}"
              f"{'...' if len(universe) > 10 else ''}")
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - args.days * 86_400_000
        # Momentum needs closes before start_ms; pad the 8h-close fetch.
        warmup_ms = mom_hours * 3_600_000
        bars_8h = math.ceil(args.days * 3) + math.ceil(warmup_ms / (8 * 3_600_000)) + 10

        histories: dict[str, SymbolHistory] = {}
        t0 = time.time()
        for i, sym in enumerate(universe, 1):
            funding, closes = await asyncio.gather(
                fetch_funding_history(b, sym, start_ms - warmup_ms, now_ms),
                fetch_perp_closes_8h(b, sym, bars_8h),
            )
            histories[sym] = SymbolHistory(funding=funding, closes_8h=closes)
            if i % 5 == 0 or i == len(universe):
                print(f"   [{i}/{len(universe)}] {sym}  ({time.time()-t0:.0f}s)",
                      flush=True)

        span_days = (now_ms - start_ms) / 86_400_000
        # NULL: plain carry on identical data.
        plain = summarise(simulate_carry(histories, start_ms=start_ms, end_ms=now_ms,
                                         p=p, start_equity=args.equity_usd,
                                         costs=costs, pit_log=pit_log),
                          start_equity=args.equity_usd, span_days=span_days)
        # OVERLAY: momentum-conditioned carry.
        overlay = summarise(simulate_carry_momentum(
            histories, start_ms=start_ms, end_ms=now_ms, p=p,
            start_equity=args.equity_usd, costs=costs,
            momentum_lookback_hours=mom_hours, pit_log=pit_log),
            start_equity=args.equity_usd, span_days=span_days)

        print("\n" + "=" * 96)
        print(f"CARRY × MOMENTUM (Card 2)  ·  {args.days}d  ·  top {p.top_n} L/S  ·  "
              f"{p.book_pct_per_side*100:.0f}%/side  ·  mom={args.momentum_days}d  ·  COSTS ON")
        print("=" * 96)
        print(_line("plain carry", plain))
        print(_line("carry×mom", overlay))
        print("-" * 96)
        beat = overlay["total_pnl_usd"] > plain["total_pnl_usd"] and \
            overlay["sharpe"] > plain["sharpe"]
        verdict = ("ALIVE — overlay beats the plain-carry null net of costs; "
                   "worth the full CPCV(10,2)+PBO grid with an 'off' config."
                   if beat else
                   "DEAD — overlay does NOT beat plain carry; the momentum mask "
                   "destroys more (breadth) than it saves. Do NOT run the CPCV grid.")
        print(f"  VERDICT: {verdict}")
        print("=" * 96)
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
