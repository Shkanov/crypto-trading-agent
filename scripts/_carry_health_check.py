"""Pre-deploy carry health-check (diagnostic, not a committed strategy).

The allocator is about to ship a carry-heavy book, but plain carry tested
NEGATIVE on the current universe (Card 1/2 runs) vs +$204 / Sharpe 0.81 on the
05-27 validated dataset. This isolates the two candidate causes:

  A) UNIVERSE-SELECTION drift — `build_universe` picks *today's* top-N by 24h
     volume and applies that fixed set across the whole past year. Volume
     leaders shift day to day, so the backtest universe is chosen with
     hindsight and its PnL is unstable to the selection date.
  B) genuine TIME decay — carry is weakening in recent windows regardless of
     universe.

Method: run the validated `simulate_carry` (unchanged) on
  - the PINNED 05-27 validated universe, and
  - today's live `build_universe`,
each over trailing 365d / 90d / 30d (same fetched histories, sliced start).
If pinned-365d ≈ +$204 but live-365d is negative ⇒ cause A (selection).
If pinned recent windows fall vs pinned-365d ⇒ cause B (decay).

Usage: BINANCE_TESTNET=false .venv/bin/python -m scripts._carry_health_check
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

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


async def _fetch(b: BinanceClient, syms: list[str], start_ms: int, now_ms: int,
                 bars_8h: int) -> dict[str, SymbolHistory]:
    out: dict[str, SymbolHistory] = {}
    for sym in syms:
        funding, closes = await asyncio.gather(
            fetch_funding_history(b, sym, start_ms, now_ms),
            fetch_perp_closes_8h(b, sym, bars_8h),
        )
        out[sym] = SymbolHistory(funding=funding, closes_8h=closes)
    return out


def _run(histories, now_ms, days, p, costs, pit_log):
    start = now_ms - days * DAY_MS
    res = simulate_carry(histories, start_ms=start, end_ms=now_ms, p=p,
                         start_equity=1_000.0, costs=costs, pit_log=pit_log)
    s = summarise(res, start_equity=1_000.0, span_days=days)
    return s


async def amain() -> None:
    load_dotenv()
    p = CarryParams()                      # validated defaults: top3, weekly, 25%/side
    costs = Costs()
    pit_log = load_pit_log(REPO / "data/research/universe/binance_delistings.json") or None

    validated_universe = json.loads(VALIDATED_JSON.read_text())["universe"]

    b = BinanceClient()
    await b.start()
    try:
        live_universe = await build_universe(b, top_n_universe=30)
        overlap = sorted(set(validated_universe) & set(live_universe))
        print(f"validated universe (n={len(validated_universe)})")
        print(f"live top-30 universe (n={len(live_universe)})")
        print(f"overlap: {len(overlap)}/30 symbols  -> {len(set(live_universe)-set(validated_universe))} changed")

        now_ms = int(time.time() * 1000)
        bars_8h = math.ceil(365 * 3) + 10
        start_365 = now_ms - 365 * DAY_MS

        print("\nfetching pinned (validated) universe ...")
        h_pin = await _fetch(b, validated_universe, start_365, now_ms, bars_8h)
        print("fetching live (today's top-30) universe ...")
        h_live = await _fetch(b, live_universe, start_365, now_ms, bars_8h)

        print("\n" + "=" * 88)
        print("CARRY HEALTH-CHECK  ·  validated params (top3 L/S, weekly, 25%/side)  ·  COSTS ON")
        print(f"  reference: 05-27 validated dataset = +$203.66 / Sharpe 0.81 / maxDD 18.9%")
        print("=" * 88)
        hdr = f"  {'universe':10s} {'window':7s} {'PnL':>10s} {'Sharpe':>7s} {'DSR':>5s} {'win':>5s} {'maxDD':>7s}"
        print(hdr); print("  " + "-" * 84)
        for tag, hh in (("pinned", h_pin), ("live-vol", h_live)):
            for days in (365, 90, 30):
                s = _run(hh, now_ms, days, p, costs, pit_log)
                print(f"  {tag:10s} {str(days)+'d':7s} {s['total_pnl_usd']:>+10.2f} "
                      f"{s['sharpe']:>+7.2f} {s['deflated_sharpe']:>+5.2f} "
                      f"{s['win_rate']*100:>4.0f}% {s['max_drawdown_pct']:>6.1f}%")
            print("  " + "-" * 84)
        print("=" * 88)
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
