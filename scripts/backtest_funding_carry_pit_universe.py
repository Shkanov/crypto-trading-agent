"""Point-in-time-universe carry re-validation (fixes selection look-ahead).

The health-check (`scripts/_carry_health_check.py`) showed the validated carry
result rests on `build_universe` applying *today's* top-N volume leaders across
the entire past year — selection look-ahead. This driver instead ranks the
universe by volume **as of each rebalance** (`trailing_quote_volume`), which is
what a live bot can actually observe, and re-runs carry without that hindsight.

To make the comparison clean it runs BOTH on the SAME fetched candidate pool
(top-`candidate-n` by *current* volume — a broad superset) and identical
price/funding histories:
  * FIXED  — today's top-`universe-n` by current volume, applied across the year
             (reproduces the look-ahead methodology), and
  * PIT    — top-`universe-n` by as-of-rebalance volume each week.
Only the universe-selection method differs. If carry's edge survives PIT, the
carry-heavy allocator deploy is justified; if it collapses, the +$204/Sharpe
0.81 validation was a selection artifact and the allocator floor needs rethink.

Residual limitation (documented honestly): the candidate pool is still drawn
from *currently-listed* names (we have no delisted-symbol klines), so this fixes
the volume-rank hindsight but not full survivorship. The PIT listing log still
drops pre-listing periods.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.backtest_funding_carry_pit_universe \\
      --days 365 --candidate-n 80 --universe-n 30 --top-n 3 --vol-window-hours 24
"""
from __future__ import annotations

import argparse
import asyncio
import math
import time
from dataclasses import dataclass, field
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
    simulate_carry,
    summarise,
)
from src.scanners.universe_pit import SymbolListing, is_active_at, load_pit_log
from src.services.costs import Costs
from src.strategies.funding_carry import (
    CarryParams,
    build_rebalance,
    cycle_pnl,
    trailing_quote_volume,
)
from src.tools.binance_client import BinanceClient

DAY_MS = 86_400_000


async def fetch_perp_klines_8h(b: BinanceClient, symbol: str, bars: int):
    """8h klines → (closes_by_close_time, quote_volume_by_close_time).
    One fetch yields both price (r[4]) and quote volume (r[7])."""
    try:
        raw = await b.fetch_klines_paginated(symbol, "8h", total=bars, market="perps")
    except Exception as e:  # noqa: BLE001
        print(f"   {symbol}: 8h kline fetch failed: {type(e).__name__}: {e}")
        return {}, {}
    closes = {int(r[6]): float(r[4]) for r in raw}
    vol = {int(r[6]): float(r[7]) for r in raw}
    return closes, vol


def simulate_carry_pit_universe(
    histories: dict[str, SymbolHistory],
    volumes: dict[str, dict[int, float]],
    start_ms: int,
    end_ms: int,
    p: CarryParams,
    start_equity: float,
    costs: Costs,
    universe_top_n: int,
    vol_window_hours: int,
    pit_log: Optional[dict[str, SymbolListing]] = None,
) -> list[WeeklyResult]:
    """Carry walk where the eligible universe at each rebalance is the top
    `universe_top_n` candidates by as-of trailing volume (PIT), then funding-
    ranked via the standard `build_rebalance`."""
    rebalance_step = p.rebalance_period_hours * 3_600_000
    results: list[WeeklyResult] = []
    equity = start_equity
    ts = start_ms

    while ts + rebalance_step <= end_ms:
        next_ts = ts + rebalance_step

        # Point-in-time volume ranking among listed candidates.
        vol_by_sym: dict[str, float] = {}
        for sym in histories:
            if pit_log is not None and not is_active_at(pit_log, sym, ts):
                continue
            v = trailing_quote_volume(volumes.get(sym, {}), ts, vol_window_hours)
            if v is not None:
                vol_by_sym[sym] = v
        eligible = sorted(vol_by_sym, key=lambda s: vol_by_sym[s], reverse=True)[:universe_top_n]

        # Funding snapshot within the as-of universe.
        snap: dict[str, float] = {}
        for sym in eligible:
            r = _funding_rate_at(histories[sym].funding, ts)
            if r is not None:
                snap[sym] = r

        rb = build_rebalance(snap, equity_usd=equity, ts_ms=ts, p=p)
        if not rb.is_active:
            ts = next_ts
            continue

        weekly_pnl = 0.0
        long_px = short_px = long_fnd = short_fnd = fee = 0.0
        for pos in rb.longs + rb.shorts:
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
            rebalance_ts_ms=ts, universe_n=len(eligible),
            longs=[x.symbol for x in rb.longs], shorts=[x.symbol for x in rb.shorts],
            long_price_pnl=long_px, short_price_pnl=short_px,
            long_funding_pnl=long_fnd, short_funding_pnl=short_fnd,
            fee_pnl=fee, total_pnl_usd=weekly_pnl, pit_filtered=0,
        ))
        ts = next_ts

    return results


def _line(tag: str, s: dict) -> str:
    return (f"  {tag:18s} PnL=${s['total_pnl_usd']:+8.2f}  Sharpe={s['sharpe']:+.2f}  "
            f"DSR={s['deflated_sharpe']:+.2f}  win={s['win_rate']*100:3.0f}%  "
            f"maxDD={s['max_drawdown_pct']:.1f}%  weeks={s['weeks']}")


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--candidate-n", type=int, default=80,
                    help="broad candidate pool (top-N by CURRENT volume) to rank within")
    ap.add_argument("--universe-n", type=int, default=30,
                    help="eligible universe size selected each rebalance")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--book-pct-per-side", type=float, default=0.25)
    ap.add_argument("--rebalance-hours", type=int, default=168)
    ap.add_argument("--vol-window-hours", type=int, default=24)
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
        print(f"PIT log: {len(pit_log)} symbols" if pit_log else "PIT DISABLED")

    p = CarryParams(top_n=args.top_n, rebalance_period_hours=args.rebalance_hours,
                    book_pct_per_side=args.book_pct_per_side)
    costs = Costs()

    b = BinanceClient()
    await b.start()
    try:
        candidates = await build_universe(b, top_n_universe=args.candidate_n)
        print(f"candidate pool: {len(candidates)} symbols (top {args.candidate_n} by current vol)")
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - args.days * DAY_MS
        bars_8h = math.ceil(args.days * 3) + 10

        histories: dict[str, SymbolHistory] = {}
        volumes: dict[str, dict[int, float]] = {}
        t0 = time.time()
        for i, sym in enumerate(candidates, 1):
            funding, (closes, vol) = await asyncio.gather(
                fetch_funding_history(b, sym, start_ms, now_ms),
                fetch_perp_klines_8h(b, sym, bars_8h),
            )
            histories[sym] = SymbolHistory(funding=funding, closes_8h=closes)
            volumes[sym] = vol
            if i % 10 == 0 or i == len(candidates):
                print(f"   [{i}/{len(candidates)}] {sym}  ({time.time()-t0:.0f}s)", flush=True)

        span_days = (now_ms - start_ms) / DAY_MS

        # FIXED: today's top universe-n (look-ahead methodology), same data.
        fixed_syms = candidates[:args.universe_n]
        fixed_hist = {s: histories[s] for s in fixed_syms}
        fixed = summarise(simulate_carry(fixed_hist, start_ms=start_ms, end_ms=now_ms,
                                         p=p, start_equity=args.equity_usd,
                                         costs=costs, pit_log=pit_log),
                          start_equity=args.equity_usd, span_days=span_days)
        # PIT: as-of-rebalance volume ranking.
        pit = summarise(simulate_carry_pit_universe(
            histories, volumes, start_ms=start_ms, end_ms=now_ms, p=p,
            start_equity=args.equity_usd, costs=costs,
            universe_top_n=args.universe_n, vol_window_hours=args.vol_window_hours,
            pit_log=pit_log),
            start_equity=args.equity_usd, span_days=span_days)

        print("\n" + "=" * 92)
        print(f"CARRY PIT-UNIVERSE RE-VALIDATION  ·  {args.days}d  ·  pool {args.candidate_n} "
              f"→ universe {args.universe_n}  ·  top {p.top_n} L/S  ·  COSTS ON")
        print(f"  reference: 05-27 fixed-universe validation = +$203.66 / Sharpe 0.81")
        print("=" * 92)
        print(_line("FIXED (look-ahead)", fixed))
        print(_line("PIT (as-of vol)", pit))
        print("-" * 92)
        survives = pit["sharpe"] > 0 and pit["total_pnl_usd"] > 0
        msg = ("carry edge SURVIVES point-in-time universe — selection look-ahead was "
               "not the whole story; carry-heavy deploy has support."
               if survives else
               "carry edge DEGRADES badly under PIT universe — the validation was a "
               "selection artifact; do NOT ship carry-heavy on the current pipeline.")
        print(f"  READ: {msg}")
        print(f"  (Δ vs FIXED: PnL {pit['total_pnl_usd']-fixed['total_pnl_usd']:+.0f}, "
              f"Sharpe {pit['sharpe']-fixed['sharpe']:+.2f})")
        print("=" * 92)
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
