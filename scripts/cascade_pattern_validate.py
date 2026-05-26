"""Validation harness for the cascade-breakout pattern detector (M2).

Gate: detector triggers within ±2 bars of his 36 actual entry timestamps
on M15 (i.e. recall + side-match measured against the corpus). Plus a
false-positive baseline: how often does the detector trigger on random
bars in the same symbol+session window?

Usage:
  .venv/bin/python -m scripts.cascade_pattern_validate
  .venv/bin/python -m scripts.cascade_pattern_validate --refetch
  .venv/bin/python -m scripts.cascade_pattern_validate --tune
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.models.types import Kline
from src.strategies.cascade_breakout import (
    CascadeParams,
    CascadePattern,
    detect_pattern,
)
from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
CALLS_PATH = REPO / "data/research/aktradescalp/aktradescalp_calls.json"
CACHE = REPO / "data/research/aktradescalp/scanner_cache"
CACHE.mkdir(parents=True, exist_ok=True)

M15_TF_MS = 15 * 60 * 1000
HISTORY_BUFFER_DAYS = 14         # 14d × 4 × 24 = 1344 M15 bars context per call
BAR_WINDOW = 2                   # ±2 bars around the call timestamp
SESSION_START_UTC = 7
SESSION_END_UTC = 12             # inclusive


# ─────────────────────────── data fetch ───────────────────────────────────


async def fetch_klines_15m(b: BinanceClient, symbol: str,
                            start_ms: int, end_ms: int, refetch: bool) -> list[list]:
    path = CACHE / f"klines_15m_{symbol}.json"
    if path.exists() and not refetch:
        return json.loads(path.read_text())
    assert b.client is not None
    out: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        async with b.rest_limiter:
            try:
                rows = await b.client.futures_klines(
                    symbol=symbol, interval="15m", limit=1000,
                    startTime=cursor, endTime=end_ms,
                )
            except Exception as e:
                msg = str(e).splitlines()[0][:80]
                print(f"  fetch err {symbol}: {msg}")
                break
        if not rows:
            break
        out.extend(rows)
        new_cursor = int(rows[-1][6]) + 1
        if new_cursor <= cursor:
            break
        cursor = new_cursor
        if len(rows) < 1000:
            break
    path.write_text(json.dumps(out))
    return out


def _row_to_kline(r: list, symbol: str) -> Kline:
    return Kline(
        symbol=symbol, timeframe="15m",
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    )


# ─────────────────────── detection at a timestamp ─────────────────────────


def _entry_bar_index(klines: list[Kline], ts_ms: int) -> Optional[int]:
    """First bar whose close_time > ts_ms — the bar a trader could close on."""
    for i, k in enumerate(klines):
        if k.close_time > ts_ms:
            return i
    return None


def _detect_in_window(klines: list[Kline], at_idx: int,
                      window: int, side: Optional[str],
                      params: CascadeParams) -> Optional[tuple[int, CascadePattern]]:
    """Try detect_pattern at every offset in [-window, +window] around at_idx.
    Returns (offset, pattern) of the closest hit matching `side` (or any side
    if side is None). Closest hit by |offset|."""
    hits: list[tuple[int, CascadePattern]] = []
    for off in range(-window, window + 1):
        i = at_idx + off
        if i < 30 or i >= len(klines):
            continue
        # Slice up to and including bar i — no look-ahead
        slice_ks = klines[: i + 1]
        pat = detect_pattern(slice_ks, params=params)
        if pat is None:
            continue
        if side is not None and pat.side != side:
            continue
        hits.append((off, pat))
    if not hits:
        return None
    hits.sort(key=lambda x: abs(x[0]))
    return hits[0]


def _random_eligible_indices(klines: list[Kline], avoid_ts_ms: int,
                              n: int, rng: random.Random) -> list[int]:
    """Pick n random bar indices in the session window, ≥1d away from
    avoid_ts_ms."""
    avoid_lo = avoid_ts_ms - 86_400_000
    avoid_hi = avoid_ts_ms + 86_400_000
    eligible = []
    for i in range(30, len(klines)):
        ct = klines[i].close_time
        if avoid_lo <= ct <= avoid_hi:
            continue
        h = datetime.fromtimestamp(ct / 1000, tz=timezone.utc).hour
        if SESSION_START_UTC <= h <= SESSION_END_UTC:
            eligible.append(i)
    if not eligible:
        return []
    return rng.sample(eligible, min(n, len(eligible)))


# ─────────────────────── driver ───────────────────────────────────────────


async def amain(refetch: bool, tune: bool) -> None:
    load_dotenv()
    calls = json.load(open(CALLS_PATH))
    calls = [c for c in calls if c["side"] in ("long", "short")]

    his_symbols = sorted({c["symbol"] for c in calls})
    print(f"loaded {len(calls)} calls, {len(his_symbols)} symbols")

    call_ts = [int(datetime.fromisoformat(c["dt_iso"]).timestamp() * 1000)
               for c in calls]
    min_ts = min(call_ts) - HISTORY_BUFFER_DAYS * 86_400_000
    max_ts = max(call_ts) + 1 * 86_400_000

    b = BinanceClient()
    await b.start()
    histories: dict[str, list[Kline]] = {}
    try:
        for i, sym in enumerate(his_symbols, 1):
            try:
                rows = await fetch_klines_15m(b, sym, min_ts, max_ts, refetch)
            except Exception as e:
                print(f"  skip {sym}: {str(e).splitlines()[0][:60]}")
                continue
            if len(rows) < 100:
                print(f"  skip {sym}: only {len(rows)} bars")
                continue
            histories[sym] = [_row_to_kline(r, sym) for r in rows]
            if i % 5 == 0:
                print(f"  fetched {i}/{len(his_symbols)}")
    finally:
        await b.close()

    print(f"\nhave M15 history for {len(histories)} / {len(his_symbols)} symbols")
    print(f"detector params: {CascadeParams()}")

    params = CascadeParams()
    if tune:
        await _tune_run(calls, histories)
        return

    # ─── Per-call detection ───
    print(f"\n=== Per-call detection (±{BAR_WINDOW} bars) ===")
    print(f"{'msg':>4s}  {'dt':19s}  {'sym':12s} {'side':5s}  {'outcome':16s}"
          f"  off  conf  R²    legs  PB   nat_c  trig_vol")
    hit_count = 0
    side_match_count = 0
    detail_rows = []
    for c in calls:
        ts = int(datetime.fromisoformat(c["dt_iso"]).timestamp() * 1000)
        sym = c["symbol"]
        side = c["side"]
        ks = histories.get(sym)
        if not ks:
            print(f"{c['msg_id']:>4d}  {c['dt_iso'][:19]}  {sym:12s} {side:5s}"
                  f"  NO_HISTORY")
            detail_rows.append({"msg_id": c["msg_id"], "outcome": "NO_HISTORY"})
            continue
        idx = _entry_bar_index(ks, ts)
        if idx is None:
            print(f"{c['msg_id']:>4d}  {c['dt_iso'][:19]}  {sym:12s} {side:5s}"
                  f"  NO_ENTRY_BAR")
            detail_rows.append({"msg_id": c["msg_id"], "outcome": "NO_ENTRY_BAR"})
            continue
        # Try side-matched first
        hit = _detect_in_window(ks, idx, BAR_WINDOW, side, params)
        outcome = "HIT_side_match" if hit else None
        if hit is None:
            # Any side
            hit = _detect_in_window(ks, idx, BAR_WINDOW, None, params)
            outcome = "HIT_wrong_side" if hit else "MISS"
        if hit is not None:
            hit_count += 1
            if hit[1].side == side:
                side_match_count += 1
            off, pat = hit
            cn = pat.cascade
            print(f"{c['msg_id']:>4d}  {c['dt_iso'][:19]}  {sym:12s} {side:5s}"
                  f"  {outcome:16s}  {off:>+2d}   {pat.confluence_count}    "
                  f"{cn.slope_r2:.2f}  {len(cn.pivots):>3d}   "
                  f"{(max(cn.pullback_ratios) if cn.pullback_ratios else 0):.2f}  "
                  f"{pat.natorgovka.compression_bars:>2d}/{pat.natorgovka.range_contraction:.2f}"
                  f"  {pat.trigger.vol_mult:.2f}")
            detail_rows.append({
                "msg_id": c["msg_id"], "outcome": outcome,
                "off": off, "det_side": pat.side, "his_side": side,
                "confluence": pat.confluence_count,
                "r2": pat.cascade.slope_r2,
                "n_pivots": len(pat.cascade.pivots),
                "max_pullback": max(pat.cascade.pullback_ratios) if pat.cascade.pullback_ratios else 0,
                "compression_bars": pat.natorgovka.compression_bars,
                "range_contraction": pat.natorgovka.range_contraction,
                "trigger_vol_mult": pat.trigger.vol_mult,
            })
        else:
            print(f"{c['msg_id']:>4d}  {c['dt_iso'][:19]}  {sym:12s} {side:5s}"
                  f"  MISS")
            detail_rows.append({"msg_id": c["msg_id"], "outcome": "MISS"})

    # ─── False-positive baseline ───
    print(f"\n=== False-positive baseline ===")
    rng = random.Random(42)
    rand_total = 0
    rand_hits = 0
    rand_per_symbol = 30
    for sym, ks in histories.items():
        for i in _random_eligible_indices(ks, avoid_ts_ms=0, n=rand_per_symbol, rng=rng):
            pat = detect_pattern(ks[: i + 1], params=params)
            rand_total += 1
            if pat is not None:
                rand_hits += 1
    fp_rate = rand_hits / rand_total if rand_total else 0
    print(f"  random bars sampled:    {rand_total}")
    print(f"  random-bar detections:  {rand_hits}  ({100*fp_rate:.1f}%)")

    # ─── Report ───
    print(f"\n=== Summary ===")
    n = len(calls)
    print(f"  total calls:               {n}")
    print(f"  detector triggered (any):  {hit_count} / {n}  "
          f"({100*hit_count/n:.1f}%)")
    print(f"  side matched his call:     {side_match_count} / {n}  "
          f"({100*side_match_count/n:.1f}%)")
    print(f"  false-positive rate:       {100*fp_rate:.1f}%")
    if rand_total > 0 and hit_count > 0:
        lift = (hit_count / n) / fp_rate if fp_rate > 0 else float("inf")
        print(f"  lift over random:          {lift:.2f}×")

    out = REPO / "data/research/aktradescalp/cascade_pattern_validation.json"
    out.write_text(json.dumps({
        "params": {k: getattr(params, k) for k in params.__dataclass_fields__},
        "calls_total": n,
        "detected_any_side": hit_count,
        "detected_correct_side": side_match_count,
        "false_positive_rate": fp_rate,
        "false_positive_n": rand_total,
        "per_call": detail_rows,
    }, indent=2))
    print(f"\nwrote {out}")


async def _tune_run(calls, histories):
    """Quick grid over key params to map sensitivity."""
    base = CascadeParams()
    grid = {
        "swing_k": [1, 2, 3],
        "cascade_min_pivots": [3, 4, 5],
        "cascade_leg_min_atr_mult": [0.5, 1.0, 1.5],
        "cascade_max_pullback": [0.5, 0.7, 0.9],
        "cascade_slope_r2_min": [0.3, 0.5, 0.7],
        "natorgovka_compression_bars_min": [2, 3, 4],
        "natorgovka_max_dist_atr": [0.3, 0.5, 0.8],
        "natorgovka_range_contraction_min": [0.1, 0.3, 0.5],
        "trigger_body_pct_min": [0.5, 0.6, 0.7],
        "trigger_vol_mult_min": [1.0, 1.3, 1.5],
    }
    print("=== Param sensitivity sweep ===")
    print(f"{'param':40s}  {'value':>8s}  recall  side")
    rng = random.Random(42)
    for pname, values in grid.items():
        for v in values:
            kwargs = {f: getattr(base, f) for f in base.__dataclass_fields__}
            kwargs[pname] = v
            p = CascadeParams(**kwargs)
            hit = 0
            side_match = 0
            for c in calls:
                ts = int(datetime.fromisoformat(c["dt_iso"]).timestamp() * 1000)
                ks = histories.get(c["symbol"])
                if not ks:
                    continue
                idx = _entry_bar_index(ks, ts)
                if idx is None:
                    continue
                h = _detect_in_window(ks, idx, BAR_WINDOW, None, p)
                if h is not None:
                    hit += 1
                    if h[1].side == c["side"]:
                        side_match += 1
            n = len(calls)
            print(f"{pname+'='+str(v):40s}  {str(v):>8s}  "
                  f"{hit}/{n}  {side_match}/{n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refetch", action="store_true")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    asyncio.run(amain(refetch=args.refetch, tune=args.tune))
