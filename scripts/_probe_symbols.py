"""Quick probe: which channel symbols are fetchable + how much history.

Usage:  .venv/bin/python -m scripts._probe_symbols
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.tools.binance_client import BinanceClient


# Channel-mentioned symbols + reference set
PROBES = [
    "FIDAUSDT", "EDENUSDT", "BSBUSDT", "PROVEUSDT", "BANANAS31USDT",
    "GRASSUSDT",   # also mentioned
    "BTCUSDT", "ETHUSDT", "SOLUSDT",  # control
]


async def amain() -> None:
    load_dotenv()
    b = BinanceClient()
    await b.start()
    try:
        for sym in PROBES:
            for market in ("spot", "perps"):
                try:
                    rows = await b.fetch_klines(sym, "1d", limit=1000, market=market)
                    if not rows:
                        print(f"{sym:14s} {market:5s}  EMPTY")
                        continue
                    first_ts = int(rows[0][6])
                    last_ts = int(rows[-1][6])
                    first = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)
                    last = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
                    days = (last_ts - first_ts) / 1000 / 86400
                    print(f"{sym:14s} {market:5s}  "
                          f"{len(rows):4d} daily bars  "
                          f"{first:%Y-%m-%d} → {last:%Y-%m-%d}  "
                          f"({days:.0f}d)")
                except Exception as e:
                    msg = str(e).split('\n')[0][:80]
                    print(f"{sym:14s} {market:5s}  ERROR: {msg}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
