"""Funding-rate monitor for Binance USDT-M perpetuals.

Polls funding every ~60s (cheap REST endpoint, no auth required) and keeps a
rolling 7-day history per symbol so the harvester can decide whether the
current rate is unusually rich vs the recent norm.

Funding rate semantics (Binance USDT-M):
- Funding payment occurs every 8h at 00:00, 08:00, 16:00 UTC.
- `lastFundingRate` is the rate that WILL be paid at the next funding time.
- Positive: longs pay shorts. Negative: shorts pay longs.
- Typical range on majors: ±5–15 bps. Extremes: ±50+ bps (capped per tier).

Annualized yield = funding_rate * 3 (payments per day) * 365.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import httpx
import structlog

from src.config.settings import get_settings

log = structlog.get_logger(__name__)

FUTURES_PREMIUM_INDEX = "https://fapi.binance.com/fapi/v1/premiumIndex"
FUTURES_FUNDING_HIST = "https://fapi.binance.com/fapi/v1/fundingRate"


@dataclass
class FundingPoint:
    symbol: str
    rate: float            # decimal, e.g. 0.0001 = 1bp
    mark_price: float
    index_price: float
    next_funding_ms: int
    ts_ms: int


@dataclass
class FundingState:
    symbol: str
    history: Deque[FundingPoint] = field(default_factory=lambda: deque(maxlen=200))
    last: Optional[FundingPoint] = None

    def current_bps(self) -> Optional[float]:
        return self.last.rate * 10_000 if self.last else None

    def avg_bps(self, n: int = 21) -> Optional[float]:
        recent = list(self.history)[-n:]
        if not recent:
            return None
        return sum(p.rate for p in recent) / len(recent) * 10_000

    def annualized_pct(self) -> Optional[float]:
        """Crude annualization: rate * 3 funding events/day * 365."""
        if self.last is None:
            return None
        return self.last.rate * 3 * 365 * 100.0


class FundingMonitor:
    """Polls funding for a fixed set of perp symbols. Run as a long-lived task."""

    def __init__(self, symbols: list[str], poll_interval_s: int = 60,
                 testnet: bool = False, binance: Optional["object"] = None) -> None:
        self.symbols = symbols
        self.poll_interval_s = poll_interval_s
        self.testnet = testnet
        # When a BinanceClient is supplied we drive live updates from the
        # `@markPrice` WS stream (zero REST weight) instead of polling
        # /premiumIndex every poll_interval_s. REST is still used for the
        # one-off historical seed. Falls back to REST polling if absent.
        self.binance = binance
        self.state: dict[str, FundingState] = {s: FundingState(symbol=s) for s in symbols}
        self._client: Optional[httpx.AsyncClient] = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    @property
    def base_url(self) -> str:
        return "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com"

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=10.0, base_url=self.base_url)
        # F15: smoke-test that the futures funding endpoint is reachable on
        # this base URL. Testnet futures has had coverage gaps; if the
        # endpoint doesn't respond cleanly we log loudly and skip polling
        # rather than silently emit None for everything.
        try:
            r = await self._client.get("/fapi/v1/premiumIndex")
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list) or not payload:
                log.warning("funding.endpoint_empty", base_url=self.base_url,
                            testnet=self.testnet)
        except Exception as e:
            log.error("funding.endpoint_unreachable", base_url=self.base_url,
                      testnet=self.testnet, err=str(e),
                      hint="funding strategy will report None — disable via FUNDING_HARVEST_ENABLED=false if persistent")
        # Seed history with last 7d so we have a baseline for "unusual" detection.
        # Also seeds CURRENT mark price into history rows (F13) by doing one
        # poll right after the historical seed.
        for sym in self.symbols:
            await self._seed_history(sym)
        await self._poll_once()
        # Backfill mark_price into the seeded rows from whatever we just polled.
        for sym, st in self.state.items():
            if st.last and st.last.mark_price > 0:
                for h in st.history:
                    if h.mark_price <= 0:
                        h.mark_price = st.last.mark_price
                        h.index_price = st.last.index_price
        # Live updates: WS mark-price stream when a client is available (no REST
        # weight), else fall back to the legacy 60s REST poll loop.
        if self.binance is not None and getattr(self.binance, "client", None) is not None:
            self._task = asyncio.create_task(self._ws_loop())
            log.info("funding.live_source", source="ws_markPrice", symbols=len(self.symbols))
        else:
            self._task = asyncio.create_task(self._poll_loop())
            log.info("funding.live_source", source="rest_poll",
                     interval_s=self.poll_interval_s)

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
        if self._client:
            await self._client.aclose()

    async def _seed_history(self, symbol: str) -> None:
        if not self._client:
            return
        try:
            r = await self._client.get("/fapi/v1/fundingRate",
                                       params={"symbol": symbol, "limit": 21})
            r.raise_for_status()
            for row in r.json():
                p = FundingPoint(
                    symbol=symbol,
                    rate=float(row["fundingRate"]),
                    mark_price=0.0, index_price=0.0,
                    next_funding_ms=int(row["fundingTime"]),
                    ts_ms=int(row["fundingTime"]),
                )
                self.state[symbol].history.append(p)
        except Exception as e:
            log.warning("funding.seed_failed", symbol=symbol, err=str(e))

    async def _poll_once(self) -> None:
        if not self._client:
            return
        try:
            r = await self._client.get("/fapi/v1/premiumIndex")
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("funding.poll_failed", err=str(e))
            return
        by_sym = {row["symbol"]: row for row in data if isinstance(data, list)}
        for sym, st in self.state.items():
            row = by_sym.get(sym)
            if not row:
                continue
            p = FundingPoint(
                symbol=sym,
                rate=float(row.get("lastFundingRate") or 0.0),
                mark_price=float(row.get("markPrice") or 0.0),
                index_price=float(row.get("indexPrice") or 0.0),
                next_funding_ms=int(row.get("nextFundingTime") or 0),
                ts_ms=int(row.get("time") or 0),
            )
            st.last = p
            # Only append on funding-time crossings to avoid duplicate history.
            if not st.history or st.history[-1].next_funding_ms != p.next_funding_ms:
                st.history.append(p)

    async def _ws_loop(self) -> None:
        """Consume the `@markPrice` WS stream and update funding state. The
        stream auto-reconnects internally; this guard only catches the
        unexpected. markPriceUpdate fields: s, p (mark), i (index), r (rate),
        T (next funding), E (event time)."""
        try:
            async for d in self.binance.stream_mark_price(self.symbols):
                sym = d.get("s")
                st = self.state.get(sym)
                if not st:
                    continue
                p = FundingPoint(
                    symbol=sym,
                    rate=float(d.get("r") or 0.0),
                    mark_price=float(d.get("p") or 0.0),
                    index_price=float(d.get("i") or 0.0),
                    next_funding_ms=int(d.get("T") or 0),
                    ts_ms=int(d.get("E") or 0),
                )
                st.last = p
                # Append to history only on funding-time crossings (avoids
                # flooding the 200-deep deque with 1/s ticks).
                if not st.history or st.history[-1].next_funding_ms != p.next_funding_ms:
                    st.history.append(p)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("funding.ws_loop.error")

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._poll_once()
            except Exception:
                log.exception("funding.poll_loop.error")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval_s)
            except asyncio.TimeoutError:
                pass

    # ----- Public read API -----
    def current(self, symbol: str) -> Optional[FundingPoint]:
        st = self.state.get(symbol)
        return st.last if st else None

    def annualized_pct(self, symbol: str) -> Optional[float]:
        st = self.state.get(symbol)
        return st.annualized_pct() if st else None

    def current_bps(self, symbol: str) -> Optional[float]:
        st = self.state.get(symbol)
        return st.current_bps() if st else None

    def avg_bps(self, symbol: str, n: int = 21) -> Optional[float]:
        st = self.state.get(symbol)
        return st.avg_bps(n) if st else None

    def is_extreme(self, symbol: str, threshold_bps: float) -> Optional[int]:
        """Returns +1 if funding > threshold (longs paying — go short perp),
        -1 if funding < -threshold (shorts paying — go long perp), 0 otherwise."""
        cur = self.current_bps(symbol)
        if cur is None:
            return None
        if cur >= threshold_bps:
            return 1
        if cur <= -threshold_bps:
            return -1
        return 0
