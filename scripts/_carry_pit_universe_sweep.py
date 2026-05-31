"""Carry universe-robustness sweep — POINT-IN-TIME version (diagnostic).

The hindsight sweep (`_carry_universe_robustness.py`) drew random 30-coin
universes from *today's* volume leaders and applied them backward — so even the
"random" draws carry survivorship/volume hindsight. This version removes that:

  - Candidate pool = top-`candidate-n` by CURRENT volume (the one residual
    hindsight we can't avoid without delisted-symbol klines — same limitation
    the committed PIT backtest documents).
  - Each draw samples a random `sub-pool` of those candidates, then runs
    `simulate_carry_pit_universe` INSIDE it: at every weekly rebalance the
    eligible universe is the top-`universe-n` by *as-of-rebalance* trailing
    volume (PIT) — what a live bot could actually see — then funding-ranked.

So the distribution answers: using only as-of-time information AND the more-legs
fix, is carry's edge robust to universe choice, or still selection luck?

Defaults bake in the more-legs fix (top-8 L/S) found to de-risk the hindsight
sweep. The deterministic full-pool PIT run is marked as a reference percentile.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts._carry_pit_universe_sweep \\
      --candidate-n 80 --sub-pool 50 --universe-n 25 --top-n 8 --draws 60 --days 365
"""
from __future__ import annotations

import argparse
import asyncio
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
    summarise,
)
from scripts.backtest_funding_carry_pit_universe import (
    fetch_perp_klines_8h,
    simulate_carry_pit_universe,
)
from src.scanners.universe_pit import SymbolListing, load_pit_log
from src.services.costs import Costs
from src.strategies.funding_carry import CarryParams
from src.tools.binance_client import BinanceClient

DAY_MS = 86_400_000


def _run_pit(hist, vols, syms, now_ms, days, p, costs, universe_n, vol_win, pit_log):
    sub_h = {s: hist[s] for s in syms if s in hist}
    sub_v = {s: vols[s] for s in syms if s in vols}
    if len(sub_h) < max(6, p.top_n * 2):
        return None
    start = now_ms - days * DAY_MS
    res = simulate_carry_pit_universe(
        sub_h, sub_v, start_ms=start, end_ms=now_ms, p=p,
        start_equity=1_000.0, costs=costs,
        universe_top_n=min(universe_n, len(sub_h)),
        vol_window_hours=vol_win, pit_log=pit_log)
    return summarise(res, start_equity=1_000.0, span_days=days)


def _pct_rank(value: float, dist: list[float]) -> float:
    return 100.0 * sum(1 for x in dist if x <= value) / len(dist)


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-n", type=int, default=80)
    ap.add_argument("--sub-pool", type=int, default=50, help="random candidate sub-pool per draw")
    ap.add_argument("--universe-n", type=int, default=25, help="PIT eligible size per rebalance")
    ap.add_argument("--top-n", type=int, default=8, help="legs per side (more-legs fix default)")
    ap.add_argument("--book-pct-per-side", type=float, default=0.25)
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--vol-window-hours", type=int, default=24)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--pit-log", default="data/research/universe/binance_delistings.json")
    args = ap.parse_args()

    random.seed(args.seed)
    pit_log: dict[str, SymbolListing] | None = None
    if args.pit_log:
        pit_log = load_pit_log(REPO / args.pit_log) or None
    p = CarryParams(top_n=args.top_n, book_pct_per_side=args.book_pct_per_side)
    costs = Costs()

    b = BinanceClient()
    await b.start()
    try:
        candidates = await build_universe(b, top_n_universe=args.candidate_n)
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - args.days * DAY_MS
        bars_8h = math.ceil(args.days * 3) + 10
        print(f"fetching {len(candidates)} candidates (funding + 8h klines) ...")
        hist: dict[str, SymbolHistory] = {}
        vols: dict[str, dict[int, float]] = {}
        t0 = time.time()
        for i, sym in enumerate(candidates, 1):
            try:
                funding, (closes, vol) = await asyncio.gather(
                    fetch_funding_history(b, sym, start_ms, now_ms),
                    fetch_perp_klines_8h(b, sym, bars_8h))
                hist[sym] = SymbolHistory(funding=funding, closes_8h=closes)
                vols[sym] = vol
            except Exception as e:  # noqa: BLE001
                print(f"  {sym}: fetch failed ({type(e).__name__})")
            if i % 10 == 0 or i == len(candidates):
                print(f"   {i}/{len(candidates)}  ({time.time()-t0:.0f}s)", flush=True)

        fetched = [s for s in candidates if s in hist]
        sub = min(args.sub_pool, len(fetched))

        pnls, sharpes = [], []
        for _ in range(args.draws):
            draw = random.sample(fetched, sub)
            s = _run_pit(hist, vols, draw, now_ms, args.days, p, costs,
                         args.universe_n, args.vol_window_hours, pit_log)
            if s is None:
                continue
            pnls.append(s["total_pnl_usd"])
            sharpes.append(s["sharpe"])

        # Reference: deterministic full-pool PIT (no random draw).
        ref = _run_pit(hist, vols, fetched, now_ms, args.days, p, costs,
                       args.universe_n, args.vol_window_hours, pit_log)

        pa, sa = np.array(pnls), np.array(sharpes)
        print("\n" + "=" * 90)
        print(f"CARRY PIT-UNIVERSE ROBUSTNESS  ·  {args.days}d  ·  {len(pnls)} random "
              f"{sub}-coin sub-pools → PIT top-{args.universe_n}/rebalance  ·  top{args.top_n} L/S  ·  COSTS ON")
        print("  (universe ranked by AS-OF-TIME volume each week — no selection hindsight)")
        print("=" * 90)
        print(f"  PnL ($1000):  mean {pa.mean():+.2f}  median {np.median(pa):+.2f}  std {pa.std():.2f}")
        print(f"                min {pa.min():+.2f}  max {pa.max():+.2f}  "
              f"p10 {np.percentile(pa,10):+.2f}  p90 {np.percentile(pa,90):+.2f}")
        print(f"  % PnL > 0:    {np.mean(pa>0)*100:.0f}%")
        print(f"  Sharpe:       mean {sa.mean():+.2f}  median {np.median(sa):+.2f}  "
              f"%>0 {np.mean(sa>0)*100:.0f}%  %>0.5 {np.mean(sa>0.5)*100:.0f}%  "
              f"%<-0.5 {np.mean(sa<-0.5)*100:.0f}%")
        if ref:
            print("  " + "-" * 86)
            rk = _pct_rank(ref["total_pnl_usd"], pnls)
            print(f"  reference (deterministic full-pool PIT): PnL {ref['total_pnl_usd']:+.2f}  "
                  f"Sharpe {ref['sharpe']:+.2f}  -> {rk:.0f}th pct")
        print("=" * 90)
        frac_pos = float(np.mean(pa > 0)); med = float(np.median(pa))
        if frac_pos >= 0.8 and med > 0:
            v = "ROBUST — edge survives PIT selection AND universe choice; deploy-worthy."
        elif frac_pos >= 0.62 and med > 0:
            v = (f"LEANING POSITIVE — {frac_pos*100:.0f}% positive, median {med:+.0f}; a real but "
                 "WEAK edge survives the PIT + more-legs corrections. Modest size only.")
        elif 0.4 <= frac_pos <= 0.62:
            v = "SELECTION-DRIVEN — still a coin flip; the edge does not survive honest PIT selection."
        else:
            v = "WEAK/FRAGILE — wide spread; regime/selection-sensitive."
        print(f"  VERDICT: {v}")
        print("=" * 90)
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
