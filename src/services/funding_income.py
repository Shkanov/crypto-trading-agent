"""Authoritative funding-income attribution via Binance's income REST endpoint.

The user-data WS ACCOUNT_UPDATE event with `m=FUNDING_FEE` carries a single
combined USDT balance delta even when MULTIPLE positions just funded — there's
no per-symbol breakdown in the WS payload. Trying to attribute by iterating
positions and crediting each ones's "share" requires guessing.

`/fapi/v1/income?incomeType=FUNDING_FEE` returns one row per (symbol, time)
with the exact income — that's the authoritative source. We poll it shortly
after each funding boundary (00:00, 08:00, 16:00 UTC + a small buffer),
dedupe by (symbol, time, income), and credit the matching open Trade.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import structlog

from src.services.storage import Storage
from src.tools.binance_client import BinanceClient

log = structlog.get_logger(__name__)

# 8h funding boundaries in seconds-since-UTC-midnight.
FUNDING_BOUNDARIES_S = [0, 8 * 3600, 16 * 3600]
# Wait this many seconds after a boundary before polling — gives Binance
# time to settle and emit income rows.
POLL_OFFSET_S = 60


class FundingIncomePoller:
    """Poll the income endpoint at each funding boundary, credit per-symbol."""

    def __init__(self, binance: BinanceClient, storage: Storage,
                 lookback_hours: int = 24) -> None:
        self.binance = binance
        self.storage = storage
        self.lookback_hours = lookback_hours
        # Dedup set: (symbol, time_ms, income_str) tuples seen this run.
        self._seen: set[tuple[str, int, str]] = set()
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        # Seed dedup with recent rows so a restart doesn't re-credit history.
        try:
            await self._poll_and_record(seed_only=True)
        except Exception:
            log.exception("funding_income.seed_failed")
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()

    def _next_poll_delay_s(self) -> int:
        now = int(time.time())
        sec_of_day = now % 86_400
        # Next boundary
        candidates = [(b + POLL_OFFSET_S) for b in FUNDING_BOUNDARIES_S]
        candidates += [c + 86_400 for c in candidates]   # tomorrow's too
        future = [c for c in candidates if c > sec_of_day]
        next_offset = min(future)
        return next_offset - sec_of_day

    async def _loop(self) -> None:
        while not self._stop.is_set():
            delay = self._next_poll_delay_s()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                return  # stop signal
            except asyncio.TimeoutError:
                pass
            try:
                await self._poll_and_record(seed_only=False)
            except Exception:
                log.exception("funding_income.poll_failed")

    async def _poll_and_record(self, seed_only: bool) -> None:
        assert self.binance.client is not None
        since_ms = int(time.time() * 1000) - self.lookback_hours * 3_600_000
        try:
            rows = await self.binance.client.futures_income_history(
                incomeType="FUNDING_FEE",
                startTime=since_ms,
                limit=1000,
            )
        except Exception as e:
            log.warning("funding_income.fetch_failed", err=str(e))
            return
        rows = rows or []
        for row in rows:
            symbol = row.get("symbol")
            t = int(row.get("time", 0))
            income_str = str(row.get("income", "0"))
            try:
                income = float(income_str)
            except (TypeError, ValueError):
                continue
            key = (symbol or "", t, income_str)
            if key in self._seen:
                continue
            self._seen.add(key)
            if seed_only:
                continue
            # Credit oldest open perp Trade for this symbol.
            open_trades = await self.storage.list_open_trades()
            candidates = [
                tr for tr in open_trades
                if tr.symbol == symbol and tr.market == "perps"
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda tr: tr.entry_ts_ms)
            target = candidates[0]
            new_total = await self.storage.credit_funding(target.id, income)
            log.info("funding_income.credited", symbol=symbol,
                     trade_id=target.id, income=income, total=new_total,
                     funding_time_ms=t)
        # Bound the dedup set — keep only the most recent ~5000 keys.
        if len(self._seen) > 5000:
            self._seen = set(list(self._seen)[-3000:])
