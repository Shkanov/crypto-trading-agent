"""Build / refresh the point-in-time listing log for Binance spot USDT pairs.

Output: data/research/universe/binance_delistings.json with shape

    {
      "BTCUSDT":  {"listed_ms": 1502942400000, "delisted_ms": null,
                   "first_kline_open_ms": ..., "source": "klines"},
      "BCCUSDT":  {"listed_ms": 1500292800000, "delisted_ms": 1573257600000,
                   "first_kline_open_ms": ..., "source": "manual"},
      ...
    }

Discovery:
  1. Query `exchangeInfo` → currently-listed USDT spot symbols. For each
     new (or stale) entry, fetch klines(interval=1d, limit=1, startTime=0)
     and take the first bar's open_time as `listed_ms`.
  2. Symbols already in the JSON that DISAPPEAR from `exchangeInfo` get
     `delisted_ms` set to the now-UTC timestamp on the run that observed
     the disappearance (overrides only if currently None). This is the
     auto-detect path; manually-set `delisted_ms` is preserved.
  3. Existing entries with both timestamps are left untouched (idempotent).

This script is conservative: it WILL NOT overwrite a manual `delisted_ms`,
and it WILL NOT replace an existing `listed_ms` that came from a prior run
(symbol listing date never changes — caching it forever is fine).

Usage:
  BINANCE_TESTNET=false .venv/bin/python -m scripts.build_pit_universe
  BINANCE_TESTNET=false .venv/bin/python -m scripts.build_pit_universe --limit 50
  BINANCE_TESTNET=false .venv/bin/python -m scripts.build_pit_universe --refresh
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.tools.binance_client import BinanceClient


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data/research/universe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "binance_delistings.json"


STABLE_BASE_TOKENS = ("FDUSD", "USDC", "EUR", "USD1", "TUSD", "BUSD", "DAI", "PYUSD",
                      "EURI", "USDP", "USTC", "USDS", "PAX", "GUSD")
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BEARUSDT", "BULLUSDT")


def _is_real_usdt_pair(symbol: str) -> bool:
    """Filter out stablecoin-vs-stablecoin pairs and leveraged tokens.
    Those skew universe stats without representing tradeable directional risk."""
    if not symbol.endswith("USDT"):
        return False
    base = symbol[:-4]
    if base in STABLE_BASE_TOKENS:
        return False
    if any(symbol.endswith(suf) for suf in LEVERAGED_SUFFIXES):
        return False
    return True


async def _first_kline_open_ms(b: BinanceClient, symbol: str) -> int | None:
    """Listed-date proxy: first daily kline's open_time. Symbols with zero
    history return None (caller should skip them)."""
    assert b.client is not None
    try:
        # startTime=0 + limit=1 returns the chronologically first bar.
        async with b.rest_limiter:
            rows = await b.client.get_klines(
                symbol=symbol, interval="1d", startTime=0, limit=1,
            )
    except Exception as e:  # noqa: BLE001
        print(f"   {symbol}: kline lookup failed: {type(e).__name__}: {e}")
        return None
    if not rows:
        return None
    return int(rows[0][0])  # open_time


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of symbols processed (smoke testing)")
    ap.add_argument("--refresh", action="store_true",
                    help="Refetch listed_ms even for already-known symbols")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan; don't write the JSON")
    args = ap.parse_args()

    existing: dict = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    print(f"existing log: {len(existing)} symbols at {OUT_PATH}")

    b = BinanceClient()
    await b.start()
    try:
        assert b.client is not None
        async with b.rest_limiter:
            info = await b.client.get_exchange_info()
        active_spot: list[str] = [
            s["symbol"] for s in info["symbols"]
            if s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and _is_real_usdt_pair(s["symbol"])
        ]
        print(f"active USDT spot universe: {len(active_spot)} symbols")

        # Auto-detect freshly-delisted: in log already, currently delisted_ms=None,
        # but no longer in active_spot. Mark with now-UTC. Manual entries with an
        # already-set delisted_ms are preserved.
        active_set = set(active_spot)
        now_ms = int(time.time() * 1000)
        autodelisted: list[str] = []
        for sym, v in existing.items():
            if (sym not in active_set
                and _is_real_usdt_pair(sym)
                and v.get("delisted_ms") is None):
                v["delisted_ms"] = now_ms
                v["source"] = v.get("source", "autodelist")
                v["autodelisted_at"] = datetime.now(timezone.utc).isoformat()
                autodelisted.append(sym)
        if autodelisted:
            print(f"auto-marked {len(autodelisted)} symbols as delisted "
                  f"(no longer in exchangeInfo): {autodelisted[:8]}"
                  f"{'…' if len(autodelisted) > 8 else ''}")

        # Decide what to fetch.
        if args.refresh:
            todo = list(active_spot)
        else:
            todo = [s for s in active_spot
                    if s not in existing or "listed_ms" not in existing[s]]
        if args.limit:
            todo = todo[: args.limit]
        print(f"to fetch listed_ms for: {len(todo)} symbols")

        if args.dry_run:
            for s in todo[:20]:
                print(f"  would fetch: {s}")
            return

        n_ok = 0
        for i, sym in enumerate(todo, 1):
            ms = await _first_kline_open_ms(b, sym)
            if ms is None:
                continue
            existing[sym] = {
                **existing.get(sym, {}),
                "listed_ms": ms,
                "first_kline_open_ms": ms,
                "delisted_ms": existing.get(sym, {}).get("delisted_ms"),
                "source": existing.get(sym, {}).get("source", "klines"),
            }
            n_ok += 1
            if i % 50 == 0 or i == len(todo):
                print(f"   [{i}/{len(todo)}] fetched (last: {sym} "
                      f"listed {datetime.fromtimestamp(ms/1000, tz=timezone.utc).date()})")

        # Persist. Sort keys for stable git-friendly output (file is gitignored,
        # but stable diffs help when copying snapshots around).
        OUT_PATH.write_text(json.dumps(dict(sorted(existing.items())), indent=2))
        print(f"\nwrote {OUT_PATH}  ({len(existing)} symbols, "
              f"{n_ok} fetched this run, {len(autodelisted)} auto-delisted)")

        # Mini-summary by listing year.
        years: dict[int, int] = {}
        for v in existing.values():
            yr = datetime.fromtimestamp(v["listed_ms"]/1000, tz=timezone.utc).year
            years[yr] = years.get(yr, 0) + 1
        print("\nlistings by year:")
        for y in sorted(years):
            print(f"   {y}: {years[y]:4d}")
        n_delisted = sum(1 for v in existing.values() if v.get("delisted_ms"))
        print(f"\ndelisted: {n_delisted} / {len(existing)}")
    finally:
        await b.close()


if __name__ == "__main__":
    asyncio.run(amain())
