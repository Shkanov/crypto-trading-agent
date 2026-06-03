"""Rebuild scanner_cache for the joint-sim / cascade validation harness.

Fetches 1h klines, 15m klines, funding, and OI history for every symbol
mentioned in aktradescalp_calls.json and saves them to
data/research/aktradescalp/scanner_cache/.

Binance retains full kline history (1h/15m go back years). OI history is
limited to ~30d by the API.

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.build_scanner_cache
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data/research/aktradescalp/scanner_cache"
CALLS = REPO / "data/research/aktradescalp/aktradescalp_calls.json"
DAYS_1H = 90
DAYS_15M = 90


async def _fetch_klines(b: BinanceClient, sym: str, tf: str, bars: int,
                        market: str = "perps") -> list:
    try:
        return await b.fetch_klines_paginated(sym, tf, total=bars, market=market)
    except Exception as e:
        print(f"   {sym} {tf}: {type(e).__name__}: {e}")
        return []


async def _fetch_funding(b: BinanceClient, sym: str,
                         start_ms: int, end_ms: int) -> list:
    out = []
    cursor = start_ms
    while cursor < end_ms:
        try:
            page = await b.client.futures_funding_rate(
                symbol=sym, startTime=cursor, endTime=end_ms, limit=1000)
        except Exception:
            break
        if not page:
            break
        out.extend(page)
        last = int(page[-1]["fundingTime"])
        if len(page) < 1000 or last <= cursor:
            break
        cursor = last + 1
    out.sort(key=lambda r: r["fundingTime"])
    return out


async def _fetch_oi(b: BinanceClient, sym: str,
                    start_ms: int, end_ms: int) -> list:
    ONE_H = 3_600_000
    out = []
    cursor = start_ms
    while cursor < end_ms:
        try:
            rows = await b.client.futures_open_interest_hist(
                symbol=sym, period="1h", limit=500,
                startTime=cursor, endTime=end_ms)
        except Exception:
            break
        if not rows:
            break
        out.extend(rows)
        last = int(rows[-1]["timestamp"])
        if last + ONE_H >= end_ms or len(rows) < 500:
            break
        cursor = last + ONE_H
    return out


async def amain() -> None:
    load_dotenv()
    CACHE.mkdir(parents=True, exist_ok=True)

    calls = json.loads(CALLS.read_text())
    symbols = sorted({c["symbol"] for c in calls if "symbol" in c})
    print(f"Rebuilding cache for {len(symbols)} symbols → {CACHE}")

    now_ms = int(time.time() * 1000)
    start_1h = now_ms - DAYS_1H * 86_400_000
    start_oi = now_ms - 29 * 86_400_000    # OI limited to ~30d

    b = BinanceClient()
    await b.start()
    try:
        # Fetch exchange info (listing dates) for PIT filtering
        exchange_info: dict[str, int] = {}
        try:
            info = await b.client.futures_exchange_info()
            for s in info.get("symbols", []):
                sym = s.get("symbol", "")
                onboard = s.get("onboardDate", 0)
                if sym and onboard:
                    exchange_info[sym] = int(onboard)
        except Exception as e:
            print(f"exchange_info fetch failed: {e}")
        (CACHE / "exchange_info.json").write_text(json.dumps(exchange_info, indent=2))
        print(f"  exchange_info: {len(exchange_info)} symbols")

        for i, sym in enumerate(symbols, 1):
            t0 = time.time()
            bars_1h = DAYS_1H * 24 + 5
            bars_15m = DAYS_15M * 96 + 10   # 96 × 15m bars per day

            k1h, k15m, funding, oi = await asyncio.gather(
                _fetch_klines(b, sym, "1h", bars_1h),
                _fetch_klines(b, sym, "15m", bars_15m),
                _fetch_funding(b, sym, start_1h, now_ms),
                _fetch_oi(b, sym, start_oi, now_ms),
            )

            (CACHE / f"klines_1h_{sym}.json").write_text(json.dumps(k1h))
            (CACHE / f"klines_15m_{sym}.json").write_text(json.dumps(k15m))
            (CACHE / f"funding_{sym}.json").write_text(json.dumps(funding))
            (CACHE / f"oi_{sym}.json").write_text(json.dumps(oi))

            print(f"  [{i}/{len(symbols)}] {sym:<16}  "
                  f"1h={len(k1h)}  15m={len(k15m)}  "
                  f"funding={len(funding)}  oi={len(oi)}  "
                  f"({time.time()-t0:.1f}s)")
    finally:
        await b.close()

    print(f"\nCache built: {CACHE}")


if __name__ == "__main__":
    asyncio.run(amain())
