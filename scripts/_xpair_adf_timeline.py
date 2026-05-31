"""Rolling-ADF cointegration decay timeline for a crypto-crypto pair.

Diagnostic for the cross-pair work (2026-05-31): the 4-pair coint sweep showed
ETH/BTC is cointegrated only ~38% of the trailing-365d bars and loses money,
despite being the 2026-05-27 sprint keeper. Question: did cointegration break at
a datable point, or has it gradually decayed?

For each rolling window (lookback_bars, stepped daily) we run the same
Engle-Granger fit + ADF test the strategy uses (`fit_engle_granger`), and track
the ADF p-value and β over time. Output: a weekly timeline table, an ASCII
sparkline of ADF-p vs the 0.05 gate, and a streak/decay summary.

This reads point-in-time windows only (each fit uses bars strictly inside its
trailing window) — it's a descriptive health timeline, not a tradeable signal.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts._xpair_adf_timeline \\
      --base ETH --quote BTC --bars 17520   # ~2y of 1h bars
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv

from src.strategies.pairs_cointegration import PairsParams, fit_engle_granger
from src.tools.binance_client import BinanceClient

SPARK = "▁▂▃▄▅▆▇█"


async def _closes(b: BinanceClient, sym: str, tf: str, bars: int):
    raw = await b.fetch_klines_paginated(sym, tf, total=bars, market="spot")
    return {int(r[6]): float(r[4]) for r in raw}


def _spark(vals, lo, hi):
    out = []
    for v in vals:
        if v is None:
            out.append(" ")
            continue
        f = (v - lo) / (hi - lo) if hi > lo else 0.0
        out.append(SPARK[min(len(SPARK) - 1, max(0, int(f * (len(SPARK) - 1))))])
    return "".join(out)


async def amain() -> None:
    load_dotenv("/Users/BulatShkanov/Downloads/crypto-trading-agent/.env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="ETH")
    ap.add_argument("--quote", default="BTC")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--bars", type=int, default=17520, help="~2y of 1h bars")
    ap.add_argument("--lookback-bars", type=int, default=24 * 60, help="60d window")
    ap.add_argument("--step-bars", type=int, default=24, help="advance 1d between fits")
    args = ap.parse_args()

    p = PairsParams(lookback_bars=args.lookback_bars)
    base_sym, quote_sym = f"{args.base}USDT", f"{args.quote}USDT"

    b = BinanceClient()
    await b.start()
    try:
        db = await _closes(b, base_sym, args.tf, args.bars)
        dq = await _closes(b, quote_sym, args.tf, args.bars)
        common = sorted(set(db) & set(dq))
        log_base = np.log(np.array([db[t] for t in common]))
        log_quote = np.log(np.array([dq[t] for t in common]))
        ts = common

        # Rolling fits stepped daily.
        rows = []  # (ts_end, adf_p, beta, is_coint)
        win = args.lookback_bars
        for i in range(win, len(ts), args.step_bars):
            fit = fit_engle_granger(log_quote[i - win:i], log_base[i - win:i], p)
            if fit is None:
                continue
            rows.append((ts[i], fit.adf_p, fit.beta, fit.is_cointegrated))

        if not rows:
            print("no fits produced — not enough history")
            return

        span0 = datetime.fromtimestamp(rows[0][0] / 1000, timezone.utc)
        span1 = datetime.fromtimestamp(rows[-1][0] / 1000, timezone.utc)
        frac = sum(r[3] for r in rows) / len(rows)
        print("\n" + "=" * 78)
        print(f"  ROLLING-ADF COINTEGRATION TIMELINE  ·  {args.base}/{quote_sym[:-4] if False else args.quote}"
              f"  ·  {args.tf}  ·  {win}-bar ({win//24}d) window, daily step")
        print(f"  {span0:%Y-%m-%d} → {span1:%Y-%m-%d}  ·  {len(rows)} fits  ·  "
              f"cointegrated (ADF p<{p.coint_pvalue_max}) {frac*100:.0f}% of windows")
        print("=" * 78)

        # Monthly aggregation.
        from collections import defaultdict
        buckets: dict[str, list] = defaultdict(list)
        for tms, adf, beta, ic in rows:
            key = datetime.fromtimestamp(tms / 1000, timezone.utc).strftime("%Y-%m")
            buckets[key].append((adf, beta, ic))
        print(f"  {'month':9s} {'med ADF-p':>10s} {'med β':>7s} {'%coint':>7s}  health")
        print("  " + "-" * 60)
        for key in sorted(buckets):
            vs = buckets[key]
            med_p = float(np.median([v[0] for v in vs]))
            med_b = float(np.median([v[1] for v in vs]))
            pct = sum(v[2] for v in vs) / len(vs) * 100
            bar = "█" * int(pct / 5)
            flag = "  <-- cointegrated" if med_p < p.coint_pvalue_max else ""
            print(f"  {key:9s} {med_p:>10.3f} {med_b:>7.2f} {pct:>6.0f}%  {bar}{flag}")

        # Sparkline of ADF-p (weekly downsample for width).
        step = max(1, len(rows) // 70)
        ws = rows[::step]
        ps = [r[1] for r in ws]
        lo, hi = 0.0, max(0.10, float(np.percentile([r[1] for r in rows], 90)))
        print("\n  ADF-p sparkline (low=cointegrated; range "
              f"{lo:.2f}–{hi:.2f}, 0.05 gate):")
        print("    " + _spark(ps, lo, hi))
        # mark where it sits vs gate
        gate_line = "".join("·" if r[1] < p.coint_pvalue_max else " " for r in ws)
        print("    " + gate_line + "   (· = below 0.05 gate)")

        # Streaks.
        longest_ok = cur_ok = longest_bad = cur_bad = 0
        ok_start = bad_start = None
        best_ok_range = best_bad_range = None
        for tms, adf, beta, ic in rows:
            if ic:
                if cur_ok == 0:
                    ok_start = tms
                cur_ok += 1
                cur_bad = 0
                if cur_ok > longest_ok:
                    longest_ok = cur_ok
                    best_ok_range = (ok_start, tms)
            else:
                if cur_bad == 0:
                    bad_start = tms
                cur_bad += 1
                cur_ok = 0
                if cur_bad > longest_bad:
                    longest_bad = cur_bad
                    best_bad_range = (bad_start, tms)
        betas = [r[2] for r in rows]
        print("\n  " + "-" * 60)

        def _fmt(rng):
            if not rng:
                return "n/a"
            a = datetime.fromtimestamp(rng[0] / 1000, timezone.utc)
            c = datetime.fromtimestamp(rng[1] / 1000, timezone.utc)
            return f"{a:%Y-%m-%d}→{c:%Y-%m-%d}"
        print(f"  longest cointegrated streak: {longest_ok} days  ({_fmt(best_ok_range)})")
        print(f"  longest broken streak:       {longest_bad} days  ({_fmt(best_bad_range)})")
        print(f"  β drift: median {np.median(betas):.2f}  range [{min(betas):.2f}, {max(betas):.2f}]"
              f"  std {np.std(betas):.2f}")
        # recent vs early
        half = len(rows) // 2
        early = sum(r[3] for r in rows[:half]) / half * 100
        late = sum(r[3] for r in rows[half:]) / (len(rows) - half) * 100
        print(f"  %coint first half: {early:.0f}%   second half: {late:.0f}%   "
              f"({'DECAYING' if late < early - 10 else 'IMPROVING' if late > early + 10 else 'stable'})")
        print("=" * 78)
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
