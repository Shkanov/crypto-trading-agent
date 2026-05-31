"""Card — OI-persistence reversal (decisive smoke).

Idea source: aktradescalp/Kovalev channel, msg 174 (2026-05-28). He longed
ESPORTSUSDT *after a sharp dump* citing "ОИ не упал, объемы — тоже" — open
interest and volume HELD through the drop. Thesis: a sharp price drop with NO
open-interest decline means positions were NOT liquidated/closed (no capitulation
or forced deleveraging) — the drop is spot-driven / absorbed, and tends to
revert. A drop WITH falling OI is genuine position unwinding and is more likely
to continue.

Decisive question (falsification-first): do dumps where OI HELD have higher
forward returns than dumps where OI FELL — and enough to beat costs when traded
long? If the OI-held cohort isn't better than the OI-fell cohort (or than buying
every dump), the filter is noise.

Data constraint: Binance `futures_open_interest_hist` serves only ~30 days, so
this is a 30-day, multi-symbol smoke — a go/no-go gate, not a CPCV-grade run. If
it survives, the next step is to capture OI live forward for a longer sample.

Method:
  1. Fetch 1h perp closes + 1h OI history for a liquid universe (~30d).
  2. Event = close dropped >= `drop_pct` over the trailing `window` bars, with a
     `window`-bar cooldown so overlapping bars don't double-count one dump.
  3. Classify each event by OI change over the same window:
       held  : OI change >= -`oi_tol`     (OI did NOT fall — the signal)
       fell  : OI change <= -`oi_fell`     (clear capitulation — the control)
  4. Event study: mean/median forward return at +4/+12/+24h for each cohort.
  5. Costed long sim of the HELD cohort: enter at event close, exit on tp/sl/time
     stop, perp taker + slippage. Verdict = held-cohort edge AND net-of-costs>0.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.backtest_oi_persistence \\
      --drop-pct 6 --window 6 --tp 6 --sl 4 --time-stop 12
"""
from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from src.services.costs import (
    Costs,
    adjust_entry_price,
    adjust_exit_price,
    impact_k_for_symbol,
    taker_fee_usd,
)
from src.tools.binance_client import BinanceClient

DEFAULT_UNIVERSE = [
    "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "ADAUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT",
    "APTUSDT", "SEIUSDT", "TIAUSDT", "WLDUSDT", "INJUSDT", "NEARUSDT",
    "FILUSDT", "PEPEUSDT",
]


@dataclass
class Bars:
    ts: np.ndarray       # close_time ms
    close: np.ndarray
    vol: np.ndarray      # quote volume per bar
    oi: np.ndarray       # open interest (contracts) aligned to ts


async def _fetch_closes(b: BinanceClient, sym: str, bars: int) -> dict[int, tuple[float, float]]:
    # Key by OPEN time (exact hour boundary) so it aligns with OI timestamps.
    # close known at end of this bar — OI for the same [open, open+1h) period is
    # likewise known then, so keying both on open_time is causal and aligned.
    raw = await b.fetch_klines_paginated(sym, "1h", total=bars, market="perps")
    return {int(r[0]): (float(r[4]), float(r[7])) for r in raw}  # open_time -> (close, qvol)


async def _fetch_oi(b: BinanceClient, sym: str, days: int) -> dict[int, float]:
    """1h OI history (last ~30d). Binance caps at 500 rows/call → page by time.
    OI `timestamp` is the period START (exact hour) — key by it directly so it
    matches the kline open_time bucket."""
    now = int(time.time() * 1000)
    start = now - days * 86_400_000
    hour = 3_600_000
    out: dict[int, float] = {}
    cursor = start
    while cursor < now:
        rows = await b.client.futures_open_interest_hist(
            symbol=sym, period="1h", limit=500,
            startTime=cursor, endTime=now)
        if not rows:
            break
        for r in rows:
            out[int(r["timestamp"])] = float(r["sumOpenInterest"])
        last = int(rows[-1]["timestamp"])
        if last + hour >= now or len(rows) < 500:
            break
        cursor = last + hour
    return out


async def fetch_bars(b: BinanceClient, sym: str, days: int) -> Optional[Bars]:
    bars = days * 24 + 5
    closes = await _fetch_closes(b, sym, bars)
    oi = await _fetch_oi(b, sym, days)
    common = sorted(set(closes) & set(oi))
    if len(common) < 48:
        return None
    return Bars(
        ts=np.array(common),
        close=np.array([closes[t][0] for t in common]),
        vol=np.array([closes[t][1] for t in common]),
        oi=np.array([oi[t] for t in common]),
    )


@dataclass
class Event:
    sym: str
    i: int
    ts: int
    drop: float          # price change over window (negative)
    oi_chg: float        # OI change over window
    vol_ratio: float     # recent vol / prior vol
    cohort: str          # "held" | "fell" | "mid"


def find_events(bars: Bars, sym: str, window: int, drop_pct: float,
                oi_tol: float, oi_fell: float) -> list[Event]:
    ev: list[Event] = []
    last_ev = -10_000
    n = len(bars.close)
    for i in range(window, n):
        if i - last_ev < window:           # cooldown: one event per dump
            continue
        c0, c1 = bars.close[i - window], bars.close[i]
        drop = (c1 - c0) / c0
        if drop > -drop_pct:               # not a big enough dump
            continue
        oi0, oi1 = bars.oi[i - window], bars.oi[i]
        oi_chg = (oi1 - oi0) / oi0 if oi0 > 0 else 0.0
        # volume: mean over the dump window vs the window before it
        prior = bars.vol[max(0, i - 2 * window):i - window]
        during = bars.vol[i - window:i]
        vol_ratio = (during.mean() / prior.mean()) if prior.size and prior.mean() > 0 else 1.0
        if oi_chg >= -oi_tol:
            cohort = "held"
        elif oi_chg <= -oi_fell:
            cohort = "fell"
        else:
            cohort = "mid"
        ev.append(Event(sym, i, int(bars.ts[i]), drop, oi_chg, vol_ratio, cohort))
        last_ev = i
    return ev


def fwd_returns(bars: Bars, ev: Event, horizons: list[int]) -> dict[int, float]:
    out = {}
    n = len(bars.close)
    for h in horizons:
        j = ev.i + h
        out[h] = (bars.close[j] - bars.close[ev.i]) / bars.close[ev.i] if j < n else np.nan
    return out


def sim_long(bars: Bars, ev: Event, tp: float, sl: float, time_stop: int,
             notional: float, costs: Costs) -> Optional[float]:
    """Long at event close; exit on tp / sl / time stop. Returns net PnL USD."""
    n = len(bars.close)
    entry_raw = bars.close[ev.i]
    ik = impact_k_for_symbol(ev.sym)
    hs = costs.half_spread_bps_default
    # crude adv: median quote-vol over trailing day, scaled to 5min
    lo = max(0, ev.i - 24)
    adv5m = float(np.median(bars.vol[lo:ev.i])) * (5.0 / 60.0) if ev.i > lo else 0.0
    entry = adjust_entry_price(entry_raw, "long", notional, adv5m, ik, hs)
    exit_raw = None
    for j in range(ev.i + 1, min(n, ev.i + time_stop + 1)):
        hi_ret = (bars.close[j] - entry) / entry
        if bars.close[j] >= entry * (1 + tp):
            exit_raw = entry * (1 + tp); break
        if bars.close[j] <= entry * (1 - sl):
            exit_raw = entry * (1 - sl); break
    if exit_raw is None:
        j = min(n - 1, ev.i + time_stop)
        exit_raw = bars.close[j]
    ex = adjust_exit_price(exit_raw, "long", notional, adv5m, ik, hs)
    gross = notional * (ex - entry) / entry
    fee = taker_fee_usd(notional, "perp", costs) * 2
    return gross - fee


def _stats(xs: list[float]) -> str:
    a = np.array([x for x in xs if not np.isnan(x)])
    if a.size == 0:
        return "n=0"
    return (f"n={a.size:3d}  mean={a.mean()*100:+.2f}%  median={np.median(a)*100:+.2f}%  "
            f"win={np.mean(a>0)*100:.0f}%")


async def amain() -> None:
    load_dotenv("/Users/BulatShkanov/Downloads/crypto-trading-agent/.env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--window", type=int, default=6, help="dump lookback bars (h)")
    ap.add_argument("--drop-pct", type=float, default=6.0, help="min drop %% over window")
    ap.add_argument("--oi-tol", type=float, default=2.0, help="HELD: OI fell <= this %%")
    ap.add_argument("--oi-fell", type=float, default=5.0, help="FELL: OI dropped >= this %%")
    ap.add_argument("--tp", type=float, default=6.0)
    ap.add_argument("--sl", type=float, default=4.0)
    ap.add_argument("--time-stop", type=int, default=12)
    ap.add_argument("--notional", type=float, default=1000.0)
    ap.add_argument("--universe", default=",".join(DEFAULT_UNIVERSE))
    args = ap.parse_args()

    drop_pct, oi_tol, oi_fell = args.drop_pct / 100, args.oi_tol / 100, args.oi_fell / 100
    tp, sl = args.tp / 100, args.sl / 100
    universe = [s.strip().upper() for s in args.universe.split(",")]
    costs = Costs()
    horizons = [4, 12, 24]

    b = BinanceClient()
    await b.start()
    try:
        all_ev: list[tuple[Event, Bars]] = []
        t0 = time.time()
        for i, sym in enumerate(universe, 1):
            try:
                bars = await fetch_bars(b, sym, args.days)
            except Exception as e:  # noqa: BLE001
                print(f"  {sym}: fetch failed ({type(e).__name__})")
                continue
            if bars is None:
                continue
            evs = find_events(bars, sym, args.window, drop_pct, oi_tol, oi_fell)
            all_ev.extend((e, bars) for e in evs)
            if i % 5 == 0 or i == len(universe):
                print(f"  [{i}/{len(universe)}] {sym}  bars={len(bars.close)}  "
                      f"events so far={len(all_ev)}  ({time.time()-t0:.0f}s)", flush=True)

        held = [(e, bars) for e, bars in all_ev if e.cohort == "held"]
        fell = [(e, bars) for e, bars in all_ev if e.cohort == "fell"]
        print("\n" + "=" * 86)
        print(f"OI-PERSISTENCE REVERSAL  ·  {args.days}d  ·  {len(universe)} syms  ·  "
              f"drop>={args.drop_pct:.0f}%/{args.window}h  ·  COSTS ON")
        print(f"  events: {len(all_ev)} total  ·  HELD(OI>=-{args.oi_tol:.0f}%)={len(held)}  "
              f"·  FELL(OI<=-{args.oi_fell:.0f}%)={len(fell)}")
        print("=" * 86)

        print("\nEVENT STUDY — forward return after the dump:")
        for label, cohort in (("HELD (signal)", held), ("FELL (control)", fell)):
            print(f"  {label}:")
            for h in horizons:
                rs = [fwd_returns(bars, e, [h])[h] for e, bars in cohort]
                print(f"    +{h:2d}h:  {_stats(rs)}")

        # Decisive: does HELD beat FELL at +12h, and does a costed HELD-long earn?
        h_dec = 12
        held_fwd = np.array([fwd_returns(bars, e, [h_dec])[h_dec] for e, bars in held])
        fell_fwd = np.array([fwd_returns(bars, e, [h_dec])[h_dec] for e, bars in fell])
        held_fwd = held_fwd[~np.isnan(held_fwd)]
        fell_fwd = fell_fwd[~np.isnan(fell_fwd)]
        edge = (held_fwd.mean() - fell_fwd.mean()) if held_fwd.size and fell_fwd.size else float("nan")

        pnls = [sim_long(bars, e, tp, sl, args.time_stop, args.notional, costs)
                for e, bars in held]
        pnls = [p for p in pnls if p is not None]
        total = sum(pnls)
        wr = np.mean([p > 0 for p in pnls]) * 100 if pnls else 0.0

        print("\n" + "-" * 86)
        print(f"DECISIVE (+{h_dec}h horizon):")
        print(f"  HELD−FELL forward-return edge: {edge*100:+.2f}%  "
              f"(held {held_fwd.mean()*100:+.2f}% vs fell {fell_fwd.mean()*100:+.2f}%)")
        print(f"  HELD-long costed sim: {len(pnls)} trades  net=${total:+.2f}  "
              f"({total/(args.notional)*100:+.1f}% on ${args.notional:.0f} notional)  win={wr:.0f}%")
        alive = (not np.isnan(edge)) and edge > 0 and total > 0
        verdict = ("ALIVE — OI-held dumps beat OI-fell AND the long earns net of costs; "
                   "worth a longer live-captured sample."
                   if alive else
                   "DEAD — OI-persistence does not separate forward returns and/or loses net of costs.")
        print(f"  VERDICT: {verdict}")
        print("=" * 86)
        if len(held) < 20 or len(fell) < 20:
            print("  ⚠ small sample (30d cap) — treat as directional, not conclusive.")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
