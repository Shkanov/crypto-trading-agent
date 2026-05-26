"""Validation harness for the aktradescalp v2 scanner.

Goal: prove (or falsify) that the cross-sectional ranking encoded in
`src/scanners/aktradescalp_scanner.py` would have surfaced his actual
36 call-symbols as top-N candidates at his actual call timestamps.

Gate to pass M1: scanner recall ≥ 40% on the 36-call corpus (i.e. his
symbol is in the scanner's top-5 at the right UTC hour) — see
`project-cascade-strategy-research` memory for the gate spec.

Approach:
  1. Build a candidate universe = his 28 symbols ∪ top-N other low-cap
     perps by current 24h vol (excluding majors).
  2. Pre-fetch (and cache to disk) for each universe symbol:
       - 1h klines covering 30d before earliest call → 1d after latest
       - funding rate history (8h cadence) over same range
       - open-interest history (1h cadence) over same range
       - exchangeInfo onboardDate (for listing-age feature)
  3. For each of his 36 calls (timestamp T, symbol S, side):
       - Build SymbolHistory for every universe symbol
       - compute_features at T for each (no look-ahead)
       - score_universe(...) → list[Candidate] sorted by score desc
       - Record: was S present? At what rank? With what side?
  4. Print per-call rows + aggregate recall@k for k=1,3,5,10, plus the
     score-distribution histogram and false-positive count.

Usage:
  .venv/bin/python -m scripts.aktradescalp_scanner_validate
  .venv/bin/python -m scripts.aktradescalp_scanner_validate --refetch
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.models.types import Kline
from src.scanners.aktradescalp_scanner import (
    Candidate,
    ScannerParams,
    SymbolFeatures,
    SymbolHistory,
    UniverseParams,
    compute_features,
    score_universe,
)
from src.tools.binance_client import BinanceClient


REPO_ROOT = Path(__file__).resolve().parent.parent
CALLS_PATH = REPO_ROOT / "data/research/aktradescalp/aktradescalp_calls.json"
CACHE_DIR = REPO_ROOT / "data/research/aktradescalp/scanner_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# History buffer before earliest call (30d for vol_z_sameHour_30d baseline)
HISTORY_BUFFER_DAYS = 32
COMPARISON_TOP_N = 50  # other low-cap alts to include for cross-sectional rank


# ───────────────────────────── data fetch ─────────────────────────────────


async def _fetch_with_cache(
    path: Path, fetcher, refetch: bool
) -> list:
    if path.exists() and not refetch:
        return json.loads(path.read_text())
    data = await fetcher()
    path.write_text(json.dumps(data))
    return data


async def fetch_klines_1h(
    b: BinanceClient, symbol: str, start_ms: int, end_ms: int, refetch: bool
) -> list[list]:
    path = CACHE_DIR / f"klines_1h_{symbol}.json"
    async def go():
        # Page back from end_ms to start_ms
        assert b.client is not None
        out: list[list] = []
        cursor = start_ms
        while cursor < end_ms:
            async with b.rest_limiter:
                rows = await b.client.futures_klines(
                    symbol=symbol, interval="1h", limit=1000,
                    startTime=cursor, endTime=end_ms,
                )
            if not rows:
                break
            out.extend(rows)
            new_cursor = int(rows[-1][6]) + 1
            if new_cursor <= cursor:
                break
            cursor = new_cursor
            if len(rows) < 1000:
                break
        return out
    return await _fetch_with_cache(path, go, refetch)


async def fetch_funding_history(
    b: BinanceClient, symbol: str, start_ms: int, end_ms: int, refetch: bool
) -> list[dict]:
    path = CACHE_DIR / f"funding_{symbol}.json"
    async def go():
        assert b.client is not None
        out: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            async with b.rest_limiter:
                rows = await b.client.futures_funding_rate(
                    symbol=symbol, startTime=cursor, endTime=end_ms, limit=1000,
                )
            if not rows:
                break
            out.extend(rows)
            new_cursor = int(rows[-1]["fundingTime"]) + 1
            if new_cursor <= cursor:
                break
            cursor = new_cursor
            if len(rows) < 1000:
                break
        return out
    return await _fetch_with_cache(path, go, refetch)


async def fetch_oi_history(
    b: BinanceClient, symbol: str, start_ms: int, end_ms: int, refetch: bool
) -> list[dict]:
    """OI history at 1h granularity. Binance caps at 500 rows/request, and
    only the last 30 days are typically retained."""
    path = CACHE_DIR / f"oi_{symbol}.json"
    async def go():
        assert b.client is not None
        out: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            try:
                async with b.rest_limiter:
                    rows = await b.client.futures_open_interest_hist(
                        symbol=symbol, period="1h", limit=500,
                        startTime=cursor, endTime=end_ms,
                    )
            except Exception as e:
                # Many alt symbols return empty OI history beyond 30d; that's fine.
                msg = str(e).splitlines()[0][:80]
                print(f"  oi_history error {symbol}: {msg}")
                break
            if not rows:
                break
            out.extend(rows)
            new_cursor = int(rows[-1]["timestamp"]) + 1
            if new_cursor <= cursor:
                break
            cursor = new_cursor
            if len(rows) < 500:
                break
        return out
    return await _fetch_with_cache(path, go, refetch)


async def fetch_exchange_info(b: BinanceClient, refetch: bool) -> dict[str, int]:
    """Returns {symbol: onboardDate_ms}. Only USDT-margined perp symbols."""
    path = CACHE_DIR / "exchange_info.json"
    if path.exists() and not refetch:
        return {k: int(v) for k, v in json.loads(path.read_text()).items()}
    assert b.client is not None
    async with b.rest_limiter:
        info = await b.client.futures_exchange_info()
    onboards: dict[str, int] = {}
    for s in info["symbols"]:
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        onboards[s["symbol"]] = int(s.get("onboardDate", 0))
    path.write_text(json.dumps(onboards))
    return onboards


async def fetch_ticker_24h(b: BinanceClient, refetch: bool) -> dict[str, float]:
    """Returns {symbol: quoteVolume24h} for all perps."""
    path = CACHE_DIR / "ticker_24h.json"
    if path.exists() and not refetch:
        return {k: float(v) for k, v in json.loads(path.read_text()).items()}
    assert b.client is not None
    async with b.rest_limiter:
        rows = await b.client.futures_ticker()
    out = {r["symbol"]: float(r["quoteVolume"]) for r in rows}
    path.write_text(json.dumps(out))
    return out


# ───────────────────────── kline row → Kline ──────────────────────────────


def _row_to_kline(r: list, symbol: str) -> Kline:
    return Kline(
        symbol=symbol, timeframe="1h",
        open_time=int(r[0]), close_time=int(r[6]),
        open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
        volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
        taker_buy_volume=float(r[9]), is_closed=True,
    )


# ───────────────────────── validation driver ──────────────────────────────


async def amain(refetch: bool, topn_report: int) -> None:
    load_dotenv()

    calls = json.load(open(CALLS_PATH))
    calls = [c for c in calls if c["side"] in ("long", "short")]
    print(f"loaded {len(calls)} calls from corpus")

    call_ts = [int(datetime.fromisoformat(c["dt_iso"]).timestamp() * 1000)
               for c in calls]
    min_ts = min(call_ts) - HISTORY_BUFFER_DAYS * 86_400_000
    max_ts = max(call_ts) + 1 * 86_400_000

    his_symbols = sorted({c["symbol"] for c in calls})
    print(f"his symbols ({len(his_symbols)}): {', '.join(his_symbols)}")

    b = BinanceClient()
    await b.start()

    try:
        # Build universe: his symbols + top-N other low-cap perps by current 24h vol
        onboards = await fetch_exchange_info(b, refetch)
        ticker = await fetch_ticker_24h(b, refetch)
        u = UniverseParams()

        comparison_pool = [
            sym for sym, vol in ticker.items()
            if sym in onboards
            and sym not in u.excluded_symbols
            and sym not in his_symbols
            and u.min_vol_24h_usd <= vol <= u.max_vol_24h_usd
        ]
        comparison_pool.sort(key=lambda s: ticker[s], reverse=True)
        comparison = comparison_pool[:COMPARISON_TOP_N]

        universe = sorted(set(his_symbols) | set(comparison))
        print(f"universe: {len(his_symbols)} his + {len(comparison)} comparison"
              f" = {len(universe)} total")

        # Pre-fetch all data
        histories: dict[str, SymbolHistory] = {}
        unavailable: list[str] = []
        for i, sym in enumerate(universe, 1):
            if sym not in onboards:
                unavailable.append(f"{sym}(no_onboard)")
                continue
            try:
                kl_rows = await fetch_klines_1h(b, sym, min_ts, max_ts, refetch)
            except Exception as e:
                unavailable.append(f"{sym}({str(e).splitlines()[0][:40]})")
                continue
            if len(kl_rows) < 30 * 24:
                unavailable.append(f"{sym}({len(kl_rows)}h)")
                continue
            funding_rows = await fetch_funding_history(b, sym, min_ts, max_ts, refetch)
            oi_rows = await fetch_oi_history(b, sym, min_ts, max_ts, refetch)

            klines = [_row_to_kline(r, sym) for r in kl_rows]
            funding = [(int(r["fundingTime"]), float(r["fundingRate"]))
                       for r in funding_rows]
            oi = [(int(r["timestamp"]), float(r["sumOpenInterest"])) for r in oi_rows]
            histories[sym] = SymbolHistory(
                symbol=sym, klines_1h=klines, funding_rates=funding,
                oi_history=oi, listing_date_ms=onboards[sym],
            )
            if i % 10 == 0:
                print(f"  fetched {i}/{len(universe)}  cached={len(histories)}"
                      f"  skipped={len(unavailable)}")

        print(f"\nready: {len(histories)} symbols with full history,"
              f" {len(unavailable)} unavailable")
        if unavailable:
            print(f"  unavailable: {', '.join(unavailable[:10])}"
                  + (f" (+{len(unavailable)-10} more)"
                     if len(unavailable) > 10 else ""))

    finally:
        await b.close()

    # ─── Run scanner over each call ───
    s = ScannerParams()
    print(f"\nscanner params: vol_z>={s.vol_z_min}  oi_z>={s.oi_z_min}"
          f"  rank_topn={s.rank_topn}  score_min={s.score_min}"
          f"  session={s.session_start_utc}-{s.session_end_utc}UTC"
          f"  fri×{s.friday_multiplier}")

    per_call_rows: list[dict] = []
    recall_hits = Counter()         # {k: count} for k in {1,3,5,10}
    out_session = 0
    no_history = 0

    for c in calls:
        ts = int(datetime.fromisoformat(c["dt_iso"]).timestamp() * 1000)
        his_sym = c["symbol"]
        his_side = c["side"]

        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        if not (s.session_start_utc <= dt.hour <= s.session_end_utc):
            out_session += 1
            per_call_rows.append({
                "msg_id": c["msg_id"], "ts_iso": c["dt_iso"],
                "symbol": his_sym, "side": his_side,
                "outcome": "OUT_OF_SESSION", "rank": None, "score": None,
            })
            continue

        if his_sym not in histories:
            no_history += 1
            per_call_rows.append({
                "msg_id": c["msg_id"], "ts_iso": c["dt_iso"],
                "symbol": his_sym, "side": his_side,
                "outcome": "NO_HISTORY", "rank": None, "score": None,
            })
            continue

        feats = {sym: compute_features(h, ts) for sym, h in histories.items()}
        ranked = score_universe(feats, ts, UniverseParams(), s)

        # Did his symbol show up? (side determination is downstream)
        rank: Optional[int] = None
        score: Optional[float] = None
        side_hint: Optional[str] = None
        for i, cand in enumerate(ranked, 1):
            if cand.symbol == his_sym:
                rank = i
                score = cand.score
                side_hint = cand.side_hint
                break

        if rank is None:
            outcome = "MISS"
        else:
            outcome = "HIT"
            for k in (1, 3, 5, 10):
                if rank <= k:
                    recall_hits[k] += 1

        per_call_rows.append({
            "msg_id": c["msg_id"], "ts_iso": c["dt_iso"],
            "symbol": his_sym, "side": his_side,
            "outcome": outcome, "rank": rank, "score": score,
            "scanner_side_hint": side_hint,
            "n_candidates": len(ranked),
        })

    # ─── Report ───
    n_eligible = len(calls) - out_session - no_history
    print(f"\n=== Per-call results (showing all {len(per_call_rows)}) ===")
    print(f"{'msg':>4s}  {'dt':19s}  {'symbol':12s} {'side':5s}  {'outcome':16s}"
          f"  rank/N   score  hint   side_match")
    for r in per_call_rows:
        rank_str = (f"{r['rank']:>3d}/{r.get('n_candidates', 0):<3d}"
                    if r["rank"] else "  -/-  ")
        score_str = f"{r['score']:.2f}" if r["score"] is not None else "  -  "
        hint = r.get("scanner_side_hint") or "-"
        side_match = "✓" if hint == r["side"] else ("✗" if hint != "-" else " ")
        print(f"{r['msg_id']:>4d}  {r['ts_iso'][:19]}  {r['symbol']:12s}"
              f" {r['side']:5s}  {r['outcome']:16s}  {rank_str}  {score_str}"
              f"  {hint:5s}  {side_match}")

    print(f"\n=== Recall summary (eligible n={n_eligible}) ===")
    print(f"  out-of-session:   {out_session:3d} / {len(calls)}")
    print(f"  no-history:       {no_history:3d} / {len(calls)}")
    print(f"  misses:           {n_eligible - recall_hits[10]:3d} / {n_eligible}")
    for k in (1, 3, 5, 10):
        hits = recall_hits[k]
        pct = 100 * hits / n_eligible if n_eligible else 0.0
        bar = "█" * int(pct / 2)
        gate = " ← GATE: ≥40%" if k == 5 else ""
        print(f"  recall@{k:<3d}  {hits:3d} / {n_eligible}  ({pct:5.1f}%) {bar}{gate}")

    # Save per-call detail
    out_path = REPO_ROOT / "data/research/aktradescalp/scanner_validation.json"
    out_path.write_text(json.dumps({
        "params": {"u": asdict(UniverseParams()), "s": asdict(s),
                   "comparison_top_n": COMPARISON_TOP_N,
                   "history_buffer_days": HISTORY_BUFFER_DAYS},
        "n_calls_total": len(calls),
        "n_out_of_session": out_session,
        "n_no_history": no_history,
        "n_eligible": n_eligible,
        "recall": {f"@{k}": recall_hits[k] for k in (1, 3, 5, 10)},
        "per_call": per_call_rows,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refetch", action="store_true",
                   help="Bust the disk cache and refetch from Binance")
    p.add_argument("--topn-report", type=int, default=5)
    args = p.parse_args()
    asyncio.run(amain(refetch=args.refetch, topn_report=args.topn_report))
