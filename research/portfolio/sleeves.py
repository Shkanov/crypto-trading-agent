"""Produce each candidate sleeve's daily-PnL series over a common window.

Honest offline measurement needs mainnet public history (testnet has none for
alts) on a FIXED calendar window so the sleeves' daily series line up for a
correlation matrix. The repo's async backtests page klines from `now` off a
testnet-bound client, so we:
  * fetch mainnet klines/funding directly (httpx, cached to /tmp/pf_cache),
  * serve them to the existing backtests via a tiny read-only shim
    (`MainnetShim.fetch_klines_paginated`), and
  * drive the sync cascade sim + a direct cross-sectional carry calc.

Every sleeve returns {utc_midnight_ms: pnl_pct_of_ref_equity} so the measurement
layer can align them. Reference equity is a fixed $1000 book per sleeve — the
allocator is correlation-driven, so the common denominator is what matters.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

CACHE = Path("/tmp/pf_cache")
CACHE.mkdir(exist_ok=True)
DAY_MS = 86_400_000
REF_EQUITY = 1000.0

_IV_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "1d": 86_400_000}


def fetch_klines(sym: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    """Paginated mainnet perps klines, cached to disk. Read-only public."""
    p = CACHE / f"k_{sym}_{interval}_{start_ms}_{end_ms}.json"
    if p.exists():
        return json.loads(p.read_text())
    rows: list[list] = []
    cur = start_ms
    with httpx.Client(base_url="https://fapi.binance.com", timeout=25.0) as cl:
        while cur < end_ms:
            r = cl.get("/fapi/v1/klines", params={
                "symbol": sym, "interval": interval, "startTime": cur,
                "endTime": end_ms, "limit": 1500})
            if r.status_code != 200:
                break
            pg = r.json()
            if not pg:
                break
            rows.extend(pg)
            last = int(pg[-1][0])
            if len(pg) < 1500 or last <= cur:
                break
            cur = last + 1
    p.write_text(json.dumps(rows))
    return rows


def fetch_funding(sym: str, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    """Mainnet funding-rate history (8h cadence), cached. Read-only public."""
    p = CACHE / f"f_{sym}_{start_ms}_{end_ms}.json"
    if p.exists():
        return [(int(t), float(v)) for t, v in json.loads(p.read_text())]
    out: list[tuple[int, float]] = []
    cur = start_ms
    with httpx.Client(base_url="https://fapi.binance.com", timeout=25.0) as cl:
        while cur < end_ms:
            r = cl.get("/fapi/v1/fundingRate", params={
                "symbol": sym, "startTime": cur, "endTime": end_ms, "limit": 1000})
            if r.status_code != 200:
                break
            pg = r.json()
            if not pg:
                break
            for row in pg:
                out.append((int(row["fundingTime"]), float(row["fundingRate"])))
            last = int(pg[-1]["fundingTime"])
            if len(pg) < 1000 or last <= cur:
                break
            cur = last + 1
    p.write_text(json.dumps(out))
    return out


def _day(ms: int) -> int:
    return (ms // DAY_MS) * DAY_MS


def daily_from_simtrades(trades, start_ms: int, end_ms: int) -> dict[int, float]:
    """Aggregate SimTrade.pnl_usd by exit UTC-day → pct of REF_EQUITY."""
    acc: dict[int, float] = {}
    for t in trades:
        if t.exit_ts_ms is None:
            continue
        d = _day(int(t.exit_ts_ms))
        if start_ms <= d < end_ms:
            acc[d] = acc.get(d, 0.0) + float(t.pnl_usd) / REF_EQUITY * 100.0
    return acc


def to_series(day_map: dict[int, float], start_ms: int, n_days: int) -> np.ndarray:
    days = [_day(start_ms) + i * DAY_MS for i in range(n_days)]
    return np.array([day_map.get(d, 0.0) for d in days], dtype=float)


# ─────────────────────────── the mainnet shim ────────────────────────────

class MainnetShim:
    """Read-only stand-in for BinanceClient that serves cached mainnet klines
    over a fixed window. Only implements what the async backtests call."""

    def __init__(self, start_ms: int, end_ms: int):
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.client = None

    async def fetch_klines_paginated(self, symbol: str, interval: str,
                                     total: int, market: str = "spot") -> list[list]:
        return fetch_klines(symbol, interval, self.start_ms, self.end_ms)

    async def fetch_klines(self, symbol: str, interval: str,
                           limit: int = 500, market: str = "spot") -> list[list]:
        return fetch_klines(symbol, interval, self.start_ms, self.end_ms)[-limit:]


# ───────────────────────────── carry sleeve ──────────────────────────────

def carry_daily(symbols: list[str], start_ms: int, end_ms: int,
                top_n: int = 3) -> dict[int, float]:
    """Cross-sectional funding carry, computed directly.

    Direction matches PRODUCTION `funding_carry.rank_for_carry` (Fan et al.
    SSRN 4666425): LONG the top_n HIGHEST-funding perps, SHORT the top_n
    LOWEST. The thesis is a PRICE bet (high-funding perps tend to outperform),
    not a funding harvest — the book actually PAYS net funding. Equal $ per leg
    ($REF_EQUITY/(2*top_n)); daily PnL = price move of the L/S legs + funding
    paid/received over the 8h hold.

    (An earlier version had longs/shorts inverted — long low / short high —
    which is the MIRROR of production and flips the sign of the result.)
    """
    # hourly closes per symbol → price at each funding time; funding history.
    closes: dict[str, dict[int, float]] = {}
    funding: dict[str, list[tuple[int, float]]] = {}
    for s in symbols:
        ks = fetch_klines(s, "1h", start_ms, end_ms)
        closes[s] = {int(r[0]): float(r[4]) for r in ks}
        funding[s] = fetch_funding(s, start_ms, end_ms)
    # funding times are aligned across symbols (00/08/16 UTC). Build the set.
    ftimes = sorted({t for s in symbols for t, _ in funding[s]})
    fmap = {s: dict(funding[s]) for s in symbols}
    leg_notional = REF_EQUITY / (2 * top_n)
    daily: dict[int, float] = {}
    for i in range(len(ftimes) - 1):
        t0, t1 = ftimes[i], ftimes[i + 1]
        rates = {s: fmap[s][t0] for s in symbols if t0 in fmap[s]}
        if len(rates) < 2 * top_n:
            continue
        order = sorted(rates, key=lambda s: rates[s])   # lowest funding first
        shorts, longs = order[:top_n], order[-top_n:]    # PROD: long HIGH, short LOW
        pnl = 0.0
        for s in longs + shorts:
            side = 1.0 if s in longs else -1.0
            # funding: a short RECEIVES funding when rate>0 → +side*(-rate)?  A
            # long PAYS funding when rate>0. Long pnl_funding = -rate*notional;
            # short = +rate*notional. So sign = -side.
            pnl += (-side) * rates[s] * leg_notional
            # price move over the hold (t0→t1), if we have both closes.
            p0 = _nearest(closes[s], t0)
            p1 = _nearest(closes[s], t1)
            if p0 and p1 and p0 > 0:
                pnl += side * (p1 - p0) / p0 * leg_notional
        d = _day(t1)
        daily[d] = daily.get(d, 0.0) + pnl / REF_EQUITY * 100.0
    return daily


def _nearest(cmap: dict[int, float], t: int) -> float:
    """Close at or just before funding time t (funding times are on the hour)."""
    for dt in (0, -3_600_000, 3_600_000, -7_200_000):
        if t + dt in cmap:
            return cmap[t + dt]
    return 0.0


# ──────────────────────────── cascade sleeve ─────────────────────────────

def cascade_daily(symbols: list[str], start_ms: int, end_ms: int) -> dict[int, float]:
    """Cascade-breakout detector on 15m mainnet klines, aggregated to daily.
    Detector-driven (no scanner gate) — a real return stream for correlation;
    the scanner selection is a separate live concern."""
    from src.models.types import Kline
    from src.services.backtest import simulate_cascade_breakout
    all_trades = []
    for s in symbols:
        raw = fetch_klines(s, "15m", start_ms, end_ms)
        if len(raw) < 100:
            continue
        ks = [Kline(symbol=s, timeframe="15m", open_time=int(r[0]),
                    close_time=int(r[6]), open=float(r[1]), high=float(r[2]),
                    low=float(r[3]), close=float(r[4]), volume=float(r[5]),
                    quote_volume=float(r[7]), trades=0, taker_buy_volume=0.0,
                    is_closed=True) for r in raw]
        try:
            _, trades = simulate_cascade_breakout(s, ks, market="perps")
            all_trades.extend(trades)
        except Exception as e:  # noqa: BLE001
            print(f"  cascade {s} failed: {e}")
    return daily_from_simtrades(all_trades, start_ms, end_ms)


# ─────────────────────────── mean-rev sleeve ─────────────────────────────

def meanrev_daily(symbols: list[str], start_ms: int, end_ms: int) -> dict[int, float]:
    """Mean-reversion backtest via the mainnet shim (5m/1h), daily-aggregated."""
    import asyncio
    from src.services.backtest import backtest_mean_reversion
    shim = MainnetShim(start_ms, end_ms)
    all_trades = []

    async def run():
        for s in symbols:
            try:
                _, trades = await backtest_mean_reversion(
                    shim, s, tf="5m", htf="1h", bars=30_000)
                all_trades.extend(trades)
            except Exception as e:  # noqa: BLE001
                print(f"  meanrev {s} failed: {e}")

    asyncio.run(run())
    return daily_from_simtrades(all_trades, start_ms, end_ms)
