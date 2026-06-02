"""Event study: extreme-funding + OI-surge directional reversal (Card 3).

Hypothesis: when a perp's funding is at a universe-relative extreme
(|funding| > 95th pctile across the universe) AND open interest is elevated
(OI z-score > threshold vs same-hour-of-day 30d baseline), the book is crowded
and fragile. Taking a position *against* the crowd (fade the longs on extreme
positive funding; fade the shorts on extreme negative funding) should capture
the mean-reversion when the crowd unwinds.

This is NOT the falsified OI-persistence reversal (backtest_oi_persistence.py),
which bet on a bounce after price drops where OI held steady. Card 3 is about
extreme *funding* marking a crowded side, confirmed by absolute OI crowding.

Structural constraint: Binance retains only ~30d of OI history. CPCV requires
≥365d. This script runs an event study: find all qualifying events in the ~30d
window, measure forward 1d and 3d directional returns, compare to a random
baseline on the same symbols.

Pass gate (per funding_edges_2026-05-29.md):
  lift over random ≥ +100bps/trade  AND  positive in each half-window.
  Fail here → stop. Do not build the full CPCV simulator without external
  OI data to extend beyond 30d.

Outputs a JSON report + prints a summary table.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.oi_extreme_event_study
  BINANCE_TESTNET=false .venv/bin/python -m scripts.oi_extreme_event_study \\
      --top-n-universe 50 --oi-z-threshold 2.0 --funding-pctile 95 \\
      --fwd-1d 3 --fwd-3d 9
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from scripts.backtest_funding_carry import (
    build_universe,
    fetch_funding_history,
    fetch_perp_closes_8h,
)
from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data/research/strategy_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EIGHT_H = 8 * 3_600_000
N_RANDOM_SEEDS = 10


# ---------------------------------------------------------------------------
# OI fetch

async def fetch_oi_1h(b: BinanceClient, sym: str,
                      start_ms: int, end_ms: int) -> dict[int, float]:
    """Fetch 1h OI history (sumOpenInterest) keyed by period open timestamp.
    Binance valid periods are up to 1h for fine granularity; 8h is not a valid
    period. We downsample to 8h alignment in _align() by matching each 8h
    bar-open to the nearest 1h OI bucket."""
    ONE_H = 3_600_000
    out: dict[int, float] = {}
    cursor = start_ms
    while cursor < end_ms:
        rows = await b.client.futures_open_interest_hist(
            symbol=sym, period="1h", limit=500,
            startTime=cursor, endTime=end_ms,
        )
        if not rows:
            break
        for r in rows:
            out[int(r["timestamp"])] = float(r["sumOpenInterest"])
        last = int(rows[-1]["timestamp"])
        if last + ONE_H >= end_ms or len(rows) < 500:
            break
        cursor = last + ONE_H
    return out


# ---------------------------------------------------------------------------
# Per-symbol aligned data

@dataclass
class SymData:
    symbol: str
    ts: list[int]           # sorted 8h timestamps
    close: list[float]
    funding: list[float]    # most-recent funding rate at each ts (may be NaN)
    oi: list[float]


def _oi_at_bar_open(ts_oi_1h: dict[int, float], bar_close_ms: int) -> Optional[float]:
    """Get OI at the open of an 8h bar given its close_time.
    Binance kline close_time = open_time + 8h - 1ms (1ms before next bar opens),
    so bar_open ≈ close_time - 8h is off by 1ms from any exact hour boundary.
    Round to the nearest 1h bucket and search ±2 neighbouring hours."""
    ONE_H = 3_600_000
    bar_open_approx = bar_close_ms - EIGHT_H
    # Floor to hour boundary (handles the -1ms offset and any minor jitter).
    hour_floor = (bar_open_approx // ONE_H) * ONE_H
    for t in (hour_floor, hour_floor + ONE_H, hour_floor - ONE_H,
              hour_floor + 2 * ONE_H, hour_floor - 2 * ONE_H):
        v = ts_oi_1h.get(t)
        if v is not None:
            return v
    return None


def _align(ts_close: dict[int, float],
           ts_funding: list[tuple[int, float]],
           ts_oi: dict[int, float],
           start_ms: int, end_ms: int) -> Optional[SymData]:
    """Build a time-aligned series on 8h timestamps in [start_ms, end_ms].
    ts_close is keyed by close_time (bar end); ts_oi is keyed by bar-open
    (1h granularity). For each close_time T, bar-open = T - 8h; OI is looked
    up at that open. Funding is filled forward (last observed rate). Bars
    missing close or OI are dropped. Returns None when fewer than 30 bars."""
    # ts_close keys are close_times (end of 8h bar); bar_open = close - 8h.
    candidates = sorted(t for t in ts_close if start_ms <= t < end_ms)
    if len(candidates) < 30:
        return None

    # Build a sorted funding list for forward-fill.
    fund_sorted = sorted(ts_funding, key=lambda x: x[0])

    ts_list, close_list, fund_list, oi_list = [], [], [], []
    fund_ptr = 0
    last_fund = float("nan")
    for t in candidates:
        oi_val = _oi_at_bar_open(ts_oi, t)
        if oi_val is None:
            continue   # no OI data for this bar — skip
        # Advance funding pointer: use the most recent rate at or before t.
        while fund_ptr < len(fund_sorted) and fund_sorted[fund_ptr][0] <= t:
            last_fund = fund_sorted[fund_ptr][1]
            fund_ptr += 1
        ts_list.append(t)
        close_list.append(ts_close[t])
        fund_list.append(last_fund)
        oi_list.append(oi_val)

    return SymData(symbol="", ts=ts_list, close=close_list,
                   funding=fund_list, oi=oi_list)


# ---------------------------------------------------------------------------
# OI z-score (same-hour-of-day 30d lookback, PIT-safe)

def _oi_zscore(oi_list: list[float], ts_list: list[int],
               idx: int, min_obs: int = 5) -> Optional[float]:
    """Z-score of OI[idx] vs same-hour-of-day values in the prior 30d window.
    Returns None when fewer than min_obs prior same-hour bars exist."""
    t = ts_list[idx]
    hour_of_day = (t // 3_600_000) % 24   # 0, 8, or 16 for 8h cadence
    window_start = t - 30 * 86_400_000
    prior = [
        oi_list[j] for j in range(idx)
        if ts_list[j] >= window_start
        and (ts_list[j] // 3_600_000) % 24 == hour_of_day
    ]
    if len(prior) < min_obs:
        return None
    mu, sigma = float(np.mean(prior)), float(np.std(prior, ddof=1))
    if sigma < 1e-12:
        return None
    return (oi_list[idx] - mu) / sigma


# ---------------------------------------------------------------------------
# Event detection + forward returns

@dataclass
class Event:
    symbol: str
    ts_ms: int
    funding_rate: float
    oi_z: float
    side: int             # +1 = long (fade negative funding), -1 = short (fade positive)
    ret_1d: Optional[float]   # directional: side * (close[t+3]/close[t] - 1)
    ret_3d: Optional[float]   # directional: side * (close[t+9]/close[t] - 1)
    half: int             # 0 = first half, 1 = second half (for sub-window check)


def find_events(
    data: dict[str, SymData],
    funding_pctile: float,
    oi_z_threshold: float,
    fwd_1d_bars: int,
    fwd_3d_bars: int,
    min_warmup_bars: int = 21,    # ~7d at 8h cadence before computing z-scores
) -> list[Event]:
    """Scan all symbols × bars for qualifying events."""
    events: list[Event] = []

    # Build per-bar universe-wide funding snapshot for cross-sectional pctile.
    # Collect all (ts, sym, funding) tuples from all symbols.
    all_ts: set[int] = set()
    for sd in data.values():
        all_ts.update(sd.ts)
    all_ts_sorted = sorted(all_ts)
    half_ts = all_ts_sorted[len(all_ts_sorted) // 2] if all_ts_sorted else 0

    # Build per-ts universe funding map: {ts: [funding_rates]}
    ts_funding_map: dict[int, list[float]] = {t: [] for t in all_ts_sorted}
    for sd in data.values():
        for i, t in enumerate(sd.ts):
            f = sd.funding[i]
            if f == f:   # not NaN
                ts_funding_map[t].append(f)

    # Precompute 95th pctile of |funding| per bar.
    ts_pctile: dict[int, Optional[float]] = {}
    for t, rates in ts_funding_map.items():
        abs_rates = [abs(r) for r in rates if r == r]
        ts_pctile[t] = (
            float(np.percentile(abs_rates, funding_pctile))
            if len(abs_rates) >= 5 else None
        )

    for sd in data.values():
        n = len(sd.ts)
        for idx in range(min_warmup_bars, n - fwd_3d_bars):
            t = sd.ts[idx]
            f = sd.funding[idx]
            if f != f:   # NaN
                continue

            # Funding extreme check.
            pctile = ts_pctile.get(t)
            if pctile is None or abs(f) <= pctile:
                continue

            # OI z-score check.
            oi_z = _oi_zscore(sd.oi, sd.ts, idx)
            if oi_z is None or oi_z <= oi_z_threshold:
                continue

            # Event found — record directional forward returns.
            side = -1 if f > 0 else +1   # fade the crowd
            c0 = sd.close[idx]
            c1d = sd.close[idx + fwd_1d_bars]
            c3d = sd.close[idx + fwd_3d_bars]
            ret_1d = side * (c1d / c0 - 1.0) if c0 > 0 else None
            ret_3d = side * (c3d / c0 - 1.0) if c0 > 0 else None

            events.append(Event(
                symbol=sd.symbol,
                ts_ms=t,
                funding_rate=f,
                oi_z=oi_z,
                side=side,
                ret_1d=ret_1d,
                ret_3d=ret_3d,
                half=0 if t < half_ts else 1,
            ))

    return events


def random_baseline(
    data: dict[str, SymData],
    n_events: int,
    fwd_1d_bars: int,
    fwd_3d_bars: int,
    min_warmup_bars: int = 21,
    seeds: int = N_RANDOM_SEEDS,
) -> dict:
    """Sample n_events random (sym, bar) pairs (no filter), compute same
    forward returns. Averaged across `seeds` random draws."""
    pool: list[tuple[str, int]] = []   # (symbol, idx)
    for sym, sd in data.items():
        n = len(sd.ts)
        pool.extend(
            (sym, i)
            for i in range(min_warmup_bars, n - fwd_3d_bars)
        )
    if not pool or n_events == 0:
        return {"ret_1d_mean": 0.0, "ret_3d_mean": 0.0, "n": 0}

    ret1_all, ret3_all = [], []
    rng = random.Random(42)
    for _ in range(seeds):
        sample = rng.sample(pool, min(n_events, len(pool)))
        for sym, idx in sample:
            sd = data[sym]
            c0 = sd.close[idx]
            if c0 <= 0:
                continue
            # Random baseline is undirected (no side) — use abs return for
            # comparison with directional event returns.
            c1d = sd.close[idx + fwd_1d_bars]
            c3d = sd.close[idx + fwd_3d_bars]
            ret1_all.append(c1d / c0 - 1.0)
            ret3_all.append(c3d / c0 - 1.0)

    return {
        "ret_1d_mean": float(np.mean(ret1_all)) if ret1_all else 0.0,
        "ret_3d_mean": float(np.mean(ret3_all)) if ret3_all else 0.0,
        "n": len(ret1_all) // seeds,
    }


# ---------------------------------------------------------------------------
# Reporting

def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean_bps": 0.0, "median_bps": 0.0,
                "win_rate": 0.0, "std_bps": 0.0}
    arr = np.array(vals) * 10_000   # convert to bps
    return {
        "n": len(vals),
        "mean_bps": float(np.mean(arr)),
        "median_bps": float(np.median(arr)),
        "win_rate": float(np.mean(arr > 0)),
        "std_bps": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def report(events: list[Event], baseline: dict,
           funding_pctile: float, oi_z_threshold: float) -> dict:
    if not events:
        return {
            "events_total": 0, "shorts": 0, "longs": 0,
            "filter": {"funding_pctile": funding_pctile, "oi_z_threshold": oi_z_threshold},
            "fwd_1d": {"all": _stats([]), "half0": _stats([]), "half1": _stats([]),
                       "shorts": _stats([]), "longs": _stats([]),
                       "baseline_bps": 0.0, "lift_bps": 0.0},
            "fwd_3d": {"all": _stats([]), "half0": _stats([]), "half1": _stats([]),
                       "shorts": _stats([]), "longs": _stats([]),
                       "baseline_bps": 0.0, "lift_bps": 0.0},
            "gate": {"lift_1d_ge_100bps": False, "positive_both_halves": False},
            "verdict": "NO EVENTS — filter too strict",
        }

    ret1 = [e.ret_1d for e in events if e.ret_1d is not None]
    ret3 = [e.ret_3d for e in events if e.ret_3d is not None]

    # Per-half sub-window.
    ret1_h0 = [e.ret_1d for e in events if e.half == 0 and e.ret_1d is not None]
    ret1_h1 = [e.ret_1d for e in events if e.half == 1 and e.ret_1d is not None]
    ret3_h0 = [e.ret_3d for e in events if e.half == 0 and e.ret_3d is not None]
    ret3_h1 = [e.ret_3d for e in events if e.half == 1 and e.ret_3d is not None]

    # Direction split.
    shorts = [e for e in events if e.side == -1]   # fading positive funding
    longs  = [e for e in events if e.side == +1]   # fading negative funding

    s1 = _stats(ret1)
    s3 = _stats(ret3)
    baseline_1d_bps = baseline["ret_1d_mean"] * 10_000
    baseline_3d_bps = baseline["ret_3d_mean"] * 10_000
    lift_1d = s1["mean_bps"] - abs(baseline_1d_bps)
    lift_3d = s3["mean_bps"] - abs(baseline_3d_bps)

    # Pass gate: lift ≥ 100bps AND positive in each half.
    half0_pos = (np.mean(ret1_h0) > 0) if ret1_h0 else False
    half1_pos = (np.mean(ret1_h1) > 0) if ret1_h1 else False
    gate_lift = lift_1d >= 100.0
    gate_halves = half0_pos and half1_pos

    if gate_lift and gate_halves:
        verdict = "PASS — proceed to full CPCV with external OI data"
    elif gate_lift and not gate_halves:
        verdict = "MARGINAL — lift clears but not consistent across sub-windows"
    else:
        verdict = "FAIL — lift < 100bps; do NOT build full simulator"

    return {
        "events_total": len(events),
        "shorts": len(shorts),
        "longs": len(longs),
        "filter": {
            "funding_pctile": funding_pctile,
            "oi_z_threshold": oi_z_threshold,
        },
        "fwd_1d": {
            "all": s1,
            "half0": _stats(ret1_h0),
            "half1": _stats(ret1_h1),
            "shorts": _stats([e.ret_1d for e in shorts if e.ret_1d is not None]),
            "longs":  _stats([e.ret_1d for e in longs  if e.ret_1d is not None]),
            "baseline_bps": baseline_1d_bps,
            "lift_bps": lift_1d,
        },
        "fwd_3d": {
            "all": s3,
            "half0": _stats(ret3_h0),
            "half1": _stats(ret3_h1),
            "shorts": _stats([e.ret_3d for e in shorts if e.ret_3d is not None]),
            "longs":  _stats([e.ret_3d for e in longs  if e.ret_3d is not None]),
            "baseline_bps": baseline_3d_bps,
            "lift_bps": lift_3d,
        },
        "gate": {
            "lift_1d_ge_100bps": gate_lift,
            "positive_both_halves": gate_halves,
        },
        "verdict": verdict,
    }


def print_report(r: dict, universe: list[str]) -> None:
    print("\n" + "=" * 90)
    print("EVENT STUDY  ·  extreme-funding + OI-surge reversal  ·  Card 3")
    print("=" * 90)
    print(f"Universe:   {len(universe)} symbols")
    print(f"Filter:     |funding| > {r['filter']['funding_pctile']}th pctile  "
          f"AND  OI z-score > {r['filter']['oi_z_threshold']}")
    print(f"Events:     {r['events_total']}  "
          f"(shorts={r['shorts']}, longs={r['longs']})")

    def _row(label: str, s: dict, baseline_bps: float, lift_bps: float) -> None:
        if s["n"] == 0:
            print(f"  {label:<12}  n=0")
            return
        print(f"  {label:<12}  n={s['n']:3d}  "
              f"mean={s['mean_bps']:+7.1f}bps  "
              f"median={s['median_bps']:+7.1f}bps  "
              f"wr={s['win_rate']*100:4.1f}%  "
              f"std={s['std_bps']:6.1f}bps  "
              f"lift_vs_random={lift_bps:+.1f}bps")

    print(f"\n1-day forward directional return  "
          f"(baseline={r['fwd_1d']['baseline_bps']:+.1f}bps):")
    f1 = r["fwd_1d"]
    _row("all",    f1["all"],    f1["baseline_bps"], f1["lift_bps"])
    _row("shorts", f1["shorts"], f1["baseline_bps"], f1["lift_bps"])
    _row("longs",  f1["longs"],  f1["baseline_bps"], f1["lift_bps"])
    _row("half-0", f1["half0"],  f1["baseline_bps"], f1["lift_bps"])
    _row("half-1", f1["half1"],  f1["baseline_bps"], f1["lift_bps"])

    print(f"\n3-day forward directional return  "
          f"(baseline={r['fwd_3d']['baseline_bps']:+.1f}bps):")
    f3 = r["fwd_3d"]
    _row("all",    f3["all"],    f3["baseline_bps"], f3["lift_bps"])
    _row("shorts", f3["shorts"], f3["baseline_bps"], f3["lift_bps"])
    _row("longs",  f3["longs"],  f3["baseline_bps"], f3["lift_bps"])
    _row("half-0", f3["half0"],  f3["baseline_bps"], f3["lift_bps"])
    _row("half-1", f3["half1"],  f3["baseline_bps"], f3["lift_bps"])

    print(f"\nPass gate:")
    print(f"  lift(1d) ≥ 100bps:         {r['gate']['lift_1d_ge_100bps']}  "
          f"({f1['lift_bps']:+.1f}bps)")
    print(f"  positive both half-windows:{r['gate']['positive_both_halves']}")
    print(f"\n  VERDICT: {r['verdict']}")


# ---------------------------------------------------------------------------
# CLI

async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=29,
                    help="OI history window (Binance caps at ~30d)")
    ap.add_argument("--top-n-universe", type=int, default=50)
    ap.add_argument("--funding-pctile", type=float, default=95.0,
                    help="Cross-sectional |funding| percentile for 'extreme' gate")
    ap.add_argument("--oi-z-threshold", type=float, default=2.0,
                    help="OI z-score threshold for 'crowded' gate")
    ap.add_argument("--fwd-1d", type=int, default=3,
                    help="Forward bars for 1d return (default 3×8h=24h)")
    ap.add_argument("--fwd-3d", type=int, default=9,
                    help="Forward bars for 3d return (default 9×8h=72h)")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    b = BinanceClient()
    await b.start()
    try:
        universe = await build_universe(b, top_n_universe=args.top_n_universe)
        print(f"universe ({len(universe)} symbols): "
              f"{', '.join(universe[:8])}{'...' if len(universe) > 8 else ''}")

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - args.days * 86_400_000
        bars_8h = args.days * 3 + 20

        print(f"Fetching {args.days}d of funding + OI + closes for "
              f"{len(universe)} symbols...")
        data: dict[str, SymData] = {}
        t0 = time.time()
        for i, sym in enumerate(universe, 1):
            funding, closes_raw, oi_raw = await asyncio.gather(
                fetch_funding_history(b, sym, start_ms, now_ms),
                fetch_perp_closes_8h(b, sym, bars_8h),
                fetch_oi_1h(b, sym, start_ms, now_ms),
            )
            sd = _align(closes_raw, funding, oi_raw, start_ms, now_ms)
            if sd is not None:
                sd.symbol = sym
                data[sym] = sd
            if i % 10 == 0 or i == len(universe):
                print(f"   [{i}/{len(universe)}]  aligned={len(data)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

        print(f"Aligned: {len(data)}/{len(universe)} symbols with ≥30 bars")
        if len(data) < 5:
            print("ERROR: too few symbols — check API access.")
            return

        print(f"\nScanning for events  "
              f"(|funding| > {args.funding_pctile}th pctile  AND  OI z > {args.oi_z_threshold})...")
        events = find_events(
            data,
            funding_pctile=args.funding_pctile,
            oi_z_threshold=args.oi_z_threshold,
            fwd_1d_bars=args.fwd_1d,
            fwd_3d_bars=args.fwd_3d,
        )
        print(f"Events found: {len(events)}")

        baseline = random_baseline(
            data, n_events=len(events),
            fwd_1d_bars=args.fwd_1d,
            fwd_3d_bars=args.fwd_3d,
        )

        r = report(events, baseline,
                   funding_pctile=args.funding_pctile,
                   oi_z_threshold=args.oi_z_threshold)
        print_report(r, universe)

        # Sensitivity: also run at relaxed thresholds if events < 20.
        if len(events) < 20:
            print(f"\n--- NOTE: only {len(events)} events — filter may be too strict. "
                  f"Re-running at relaxed thresholds (pctile=90, oi_z=1.5)...")
            events2 = find_events(
                data, funding_pctile=90.0, oi_z_threshold=1.5,
                fwd_1d_bars=args.fwd_1d, fwd_3d_bars=args.fwd_3d,
            )
            baseline2 = random_baseline(
                data, n_events=len(events2),
                fwd_1d_bars=args.fwd_1d, fwd_3d_bars=args.fwd_3d,
            )
            r2 = report(events2, baseline2, 90.0, 1.5)
            print(f"Relaxed events: {len(events2)}")
            print_report(r2, universe)
            out_data = {"strict": r, "relaxed": r2,
                        "events_strict": [asdict(e) for e in events],
                        "events_relaxed": [asdict(e) for e in events2]}
        else:
            out_data = {"strict": r,
                        "events": [asdict(e) for e in events]}

        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.out_tag}" if args.out_tag else ""
        out_path = OUT_DIR / f"oi_extreme_event_study_{ts_tag}{tag}.json"
        out_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "days": args.days,
            "universe": universe,
            "n_aligned": len(data),
            **out_data,
        }, indent=2, default=str))
        print(f"\nwrote {out_path}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
