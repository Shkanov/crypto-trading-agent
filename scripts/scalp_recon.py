"""Scalping recon — sample perps bookTicker for spread/vol distribution.

Step 1 of the scalping scope: before building anything, prove the typical
bid-ask spread on candidate symbols is wide enough relative to the maker
fee to leave room for edge. If it isn't, scalping is dead on these symbols
and we save weeks of work.

Outputs:
  - data/scalp_recon.jsonl   raw ticks (one line per bookTicker update)
  - periodic stdout summary  spread mean / p50 / p95 / p99, fraction of
                             time spread exceeds 2x maker fee, realized
                             vol per minute from midprice.

Usage:
  python -m scripts.scalp_recon --symbols BTCUSDT,ETHUSDT,SOLUSDT --duration-min 10
  python -m scripts.scalp_recon --symbols DOGEUSDT,AVAXUSDT --duration-min 30 \
      --out data/scalp_recon_alts.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import websockets

from src.config.settings import get_settings


FUTURES_WS_BASE = "wss://fstream.binance.com/stream"


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
    return sorted_vals[idx]


def _vol_per_minute_bps(midprices_by_min: dict[int, list[float]]) -> float:
    mins = sorted(midprices_by_min.keys())
    if len(mins) < 2:
        return 0.0
    closes = [midprices_by_min[m][-1] for m in mins]
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    if len(rets) < 2:
        return abs(rets[0]) * 10_000 if rets else 0.0
    return statistics.stdev(rets) * 10_000


def _print_summary(spreads_bps: dict[str, list[float]],
                   midprices_by_min: dict[str, dict[int, list[float]]],
                   maker_fee_bps: float, started: float, n_ticks: int,
                   final: bool = False) -> None:
    label = "FINAL" if final else "ROLLING"
    elapsed = time.time() - started
    print(f"\n=== {label} after {elapsed:.0f}s, n_ticks={n_ticks} ===")
    threshold_2x = 2 * maker_fee_bps
    threshold_4x = 4 * maker_fee_bps
    for sym in sorted(spreads_bps.keys()):
        vals = spreads_bps[sym]
        if not vals:
            print(f"  {sym}: no ticks")
            continue
        s = sorted(vals)
        n = len(s)
        mean_s = sum(s) / n
        p50 = _percentile(s, 0.50)
        p95 = _percentile(s, 0.95)
        p99 = _percentile(s, 0.99)
        wide_2x = sum(1 for x in vals if x > threshold_2x) / n
        wide_4x = sum(1 for x in vals if x > threshold_4x) / n
        vol_pm = _vol_per_minute_bps(midprices_by_min[sym])
        print(f"  {sym}: ticks={n:>6}  spread mean={mean_s:5.2f}  "
              f"p50={p50:5.2f}  p95={p95:5.2f}  p99={p99:6.2f}  "
              f">{threshold_2x:.0f}bps={wide_2x:5.1%}  "
              f">{threshold_4x:.0f}bps={wide_4x:5.1%}  vol/min={vol_pm:5.1f}bps")


async def amain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--duration-min", type=float, default=10.0)
    ap.add_argument("--out", default="data/scalp_recon.jsonl")
    ap.add_argument("--print-every-sec", type=int, default=30)
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    duration_sec = args.duration_min * 60.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    maker_fee_bps = settings.perps_maker_fee_bps

    # Direct WS — python-binance's futures_multiplex_socket silently fails
    # to forward messages in some versions; raw WS is simpler and works.
    streams_param = "/".join(f"{sym.lower()}@bookTicker" for sym in symbols)
    url = f"{FUTURES_WS_BASE}?streams={streams_param}"

    spreads_bps: dict[str, list[float]] = defaultdict(list)
    midprices_by_min: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    started = time.time()
    last_print = started
    n_ticks = 0

    print(f"recording {symbols} perps bookTicker for {args.duration_min:.1f} min "
          f"-> {out_path}  (maker_fee_bps={maker_fee_bps})")

    try:
        with out_path.open("w") as f:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                while time.time() - started < duration_sec:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        continue
                    msg = json.loads(raw)
                    data = msg.get("data", msg)
                    sym = data.get("s")
                    if not sym:
                        continue
                    try:
                        bid = float(data["b"])
                        ask = float(data["a"])
                        bid_qty = float(data["B"])
                        ask_qty = float(data["A"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if bid <= 0 or ask <= 0 or ask < bid:
                        continue
                    mid = (bid + ask) / 2
                    spread_bps = (ask - bid) / mid * 10_000
                    ts_ms = int(data.get("E") or data.get("T") or time.time() * 1000)
                    minute = ts_ms // 60_000

                    spreads_bps[sym].append(spread_bps)
                    midprices_by_min[sym][minute].append(mid)
                    n_ticks += 1

                    f.write(json.dumps({
                        "ts_ms": ts_ms, "symbol": sym,
                        "bid": bid, "ask": ask,
                        "bid_qty": bid_qty, "ask_qty": ask_qty,
                        "spread_bps": spread_bps,
                    }) + "\n")

                    now = time.time()
                    if now - last_print >= args.print_every_sec:
                        _print_summary(spreads_bps, midprices_by_min,
                                       maker_fee_bps, started, n_ticks, final=False)
                        last_print = now
    finally:
        _print_summary(spreads_bps, midprices_by_min,
                       maker_fee_bps, started, n_ticks, final=True)


if __name__ == "__main__":
    asyncio.run(amain())
