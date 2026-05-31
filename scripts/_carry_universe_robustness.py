"""Carry universe-robustness sweep (diagnostic, not a committed strategy).

The carry health-check (2026-05-31) showed the sign of 365d carry PnL FLIPS
between the pinned-05-27 universe (−$120) and today's live top-30 (+$71). That
means the result may be driven by *which* 30 coins you picked, not by the carry
signal. `build_universe` picks today's volume leaders and applies them backward
(hindsight selection), so a single universe choice is one draw from a wide
distribution.

This quantifies that: draw many RANDOM 30-coin universes from a larger volume
pool, run the validated `simulate_carry` (unchanged, costs ON, PIT) on each over
the same window, and report the PnL/Sharpe DISTRIBUTION. Reading:
  - If most draws are positive and tightly clustered ⇒ the edge is in the signal.
  - If draws straddle zero with a wide spread ⇒ the "edge" is mostly which coins
    you happened to pick (selection luck), and the validated single-universe
    PASS is not trustworthy.

The pinned-05-27 and live top-30 universes are marked as reference points so you
can see where the two "official" choices fall in the luck distribution.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts._carry_universe_robustness \\
      --pool 60 --size 30 --draws 60 --days 365
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time

import numpy as np
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
from src.scanners.universe_pit import load_pit_log
from src.services.costs import Costs
from src.strategies.funding_carry import CarryParams
from src.tools.binance_client import BinanceClient

VALIDATED_JSON = REPO / "data/research/strategy_tuning/funding_carry_20260527_124532_pit.json"
DAY_MS = 86_400_000


async def _fetch(b, syms, start_ms, now_ms, bars_8h):
    out: dict[str, SymbolHistory] = {}
    t0 = time.time()
    for i, sym in enumerate(syms, 1):
        try:
            funding, closes = await asyncio.gather(
                fetch_funding_history(b, sym, start_ms, now_ms),
                fetch_perp_closes_8h(b, sym, bars_8h),
            )
            out[sym] = SymbolHistory(funding=funding, closes_8h=closes)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: fetch failed ({type(e).__name__})")
        if i % 10 == 0 or i == len(syms):
            print(f"   fetched {i}/{len(syms)}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def _run_universe(histories, syms, now_ms, days, p, costs, pit_log):
    sub = {s: histories[s] for s in syms if s in histories}
    if len(sub) < 6:
        return None
    start = now_ms - days * DAY_MS
    res = simulate_carry(sub, start_ms=start, end_ms=now_ms, p=p,
                         start_equity=1_000.0, costs=costs, pit_log=pit_log)
    return summarise(res, start_equity=1_000.0, span_days=days)


def _pct_rank(value: float, dist: list[float]) -> float:
    return 100.0 * sum(1 for x in dist if x <= value) / len(dist)


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=60, help="candidate pool size (top-N by vol)")
    ap.add_argument("--size", type=int, default=30, help="universe size per draw")
    ap.add_argument("--draws", type=int, default=60, help="number of random universes")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--top-n", type=int, default=3,
                    help="legs per side (the 'more-legs' diversification knob)")
    ap.add_argument("--book-pct-per-side", type=float, default=0.25)
    args = ap.parse_args()

    random.seed(args.seed)
    p = CarryParams(top_n=args.top_n, book_pct_per_side=args.book_pct_per_side)
    costs = Costs()
    pit_log = load_pit_log(REPO / "data/research/universe/binance_delistings.json") or None
    validated_universe = json.loads(VALIDATED_JSON.read_text())["universe"]

    b = BinanceClient()
    await b.start()
    try:
        pool = await build_universe(b, top_n_universe=args.pool)
        live30 = pool[:30]
        # Pool must also cover the pinned universe so it can be scored on the
        # same fetched histories.
        all_syms = sorted(set(pool) | set(validated_universe))
        print(f"pool top-{args.pool} ({len(pool)}) + pinned extras "
              f"-> {len(all_syms)} symbols to fetch")

        now_ms = int(time.time() * 1000)
        start_365 = now_ms - args.days * DAY_MS
        bars_8h = math.ceil(args.days * 3) + 10
        print("fetching histories ...")
        hist = await _fetch(b, all_syms, start_365, now_ms, bars_8h)

        # Random draws from the pool (only symbols we actually fetched).
        fetched_pool = [s for s in pool if s in hist]
        size = min(args.size, len(fetched_pool))
        pnls, sharpes, dsrs, dds = [], [], [], []
        for _ in range(args.draws):
            draw = random.sample(fetched_pool, size)
            s = _run_universe(hist, draw, now_ms, args.days, p, costs, pit_log)
            if s is None:
                continue
            pnls.append(s["total_pnl_usd"])
            sharpes.append(s["sharpe"])
            dsrs.append(s["deflated_sharpe"])
            dds.append(s["max_drawdown_pct"])

        # Reference universes on the same fetched histories.
        ref = {}
        for tag, syms in (("pinned-0527", validated_universe), ("live-top30", live30)):
            s = _run_universe(hist, syms, now_ms, args.days, p, costs, pit_log)
            if s:
                ref[tag] = s

        pa = np.array(pnls)
        sa = np.array(sharpes)
        print("\n" + "=" * 88)
        print(f"CARRY UNIVERSE-ROBUSTNESS  ·  {args.days}d  ·  {len(pnls)} random "
              f"{size}-coin universes from top-{args.pool}  ·  top{args.top_n} L/S  ·  COSTS ON")
        print("=" * 88)
        print(f"  PnL ($1000 book):  mean {pa.mean():+.2f}  median {np.median(pa):+.2f}  "
              f"std {pa.std():.2f}")
        print(f"                     min {pa.min():+.2f}  max {pa.max():+.2f}  "
              f"p10 {np.percentile(pa,10):+.2f}  p90 {np.percentile(pa,90):+.2f}")
        print(f"  % PnL > 0:         {np.mean(pa>0)*100:.0f}%")
        print(f"  Sharpe:            mean {sa.mean():+.2f}  median {np.median(sa):+.2f}  "
              f"% Sharpe>0: {np.mean(sa>0)*100:.0f}%")
        print(f"  Sharpe>0.5 (size-able): {np.mean(sa>0.5)*100:.0f}%   "
              f"Sharpe<-0.5 (clearly bad): {np.mean(sa<-0.5)*100:.0f}%")
        print("  " + "-" * 84)
        print("  reference universes (where the two 'official' choices land):")
        for tag, s in ref.items():
            rk = _pct_rank(s["total_pnl_usd"], pnls)
            print(f"    {tag:12s}  PnL {s['total_pnl_usd']:+8.2f}  Sharpe {s['sharpe']:+.2f}  "
                  f"-> {rk:.0f}th pct of the luck distribution")
        print("=" * 88)
        # Verdict heuristic.
        frac_pos = float(np.mean(pa > 0))
        med = float(np.median(pa))
        if frac_pos >= 0.8 and med > 0:
            verdict = "ROBUST — most universes positive; edge is in the signal, not selection."
        elif frac_pos >= 0.62 and med > 0:
            verdict = (f"LEANING POSITIVE — {frac_pos*100:.0f}% of universes positive, median "
                       f"{med:+.0f}; a real but WEAK edge survives selection noise. Deployable at "
                       "modest size with this many legs; not high-conviction.")
        elif 0.4 <= frac_pos <= 0.62 and pa.min() < 0 < pa.max():
            verdict = ("SELECTION-DRIVEN — draws straddle zero ~50/50; the 'edge' is mostly "
                       "which coins you picked, NOT a reliable signal. Single-universe PASS is luck.")
        else:
            verdict = ("WEAK/FRAGILE — leans one way but wide spread; carry is regime/"
                       "selection-sensitive, size with caution.")
        print(f"  VERDICT: {verdict}")
        print("=" * 88)
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
