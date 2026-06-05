"""Tiny on-disk cache for warmup klines.

Boot warmup fetches `len(symbols) × len(timeframes)` kline pages over REST.
On a normal weekly reboot that's a one-off and fine; during a rapid restart
(crash loop, operator bouncing the service) it's a REST burst that, stacked,
contributes to a -1003 IP ban. Caching the last warmup payload to disk lets a
restart within `ttl_s` reuse it and issue ZERO warmup REST calls.

Freshness tradeoff: a cache hit seeds indicators with klines up to `ttl_s`
old; the live WS stream takes over going forward. Kept short by default, and
the only live strategy (Δfunding) doesn't use these klines at all — they feed
the (disabled) indicator strategy + anomaly detectors. A stale-but-recent seed
is harmless there; rolling indicator windows heal any small gap.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

CACHE_DIR = Path("./data/cache/klines")


def _path(market: str, symbol: str, timeframe: str) -> Path:
    return CACHE_DIR / f"{market}_{symbol}_{timeframe}.json"


def load(market: str, symbol: str, timeframe: str, ttl_s: float) -> Optional[list]:
    """Return cached raw klines if the cache exists and is younger than ttl_s,
    else None. Never raises — a corrupt/missing cache is treated as a miss."""
    p = _path(market, symbol, timeframe)
    try:
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > ttl_s:
            return None
        with p.open() as f:
            payload = json.load(f)
        raw = payload.get("raw")
        return raw if isinstance(raw, list) and raw else None
    except Exception as e:  # noqa: BLE001
        log.warning("kline_cache.load_failed", symbol=symbol, tf=timeframe, err=str(e))
        return None


def save(market: str, symbol: str, timeframe: str, raw: list) -> None:
    """Persist raw klines. Never raises — caching is best-effort."""
    p = _path(market, symbol, timeframe)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump({"fetched_ms": int(time.time() * 1000), "raw": raw}, f)
        tmp.replace(p)  # atomic
    except Exception as e:  # noqa: BLE001
        log.warning("kline_cache.save_failed", symbol=symbol, tf=timeframe, err=str(e))
