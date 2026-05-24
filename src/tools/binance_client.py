"""Binance REST + WebSocket client wrapper.

Wraps `python-binance.AsyncClient` + `BinanceSocketManager` with:
- testnet/mainnet switching for spot AND USD-M futures
- exchangeInfo cache + filter-aware quantization (Decimal)
- per-scope rate limiters (weight + orders) driven by response headers
- listenKey lifecycle and reconnection supervisors
- deterministic clientOrderId hand-off to executor (set by caller)

Hot loop does not call into REST directly; the executor does.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any, AsyncIterator, Optional

import structlog
from aiolimiter import AsyncLimiter
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException

from src.config.settings import get_settings

log = structlog.get_logger(__name__)


@dataclass
class SymbolFilters:
    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_notional: Decimal
    base_asset: str
    quote_asset: str


def _floor_to_step(value: float, step: Decimal) -> Decimal:
    if step <= 0:
        return Decimal(str(value))
    d = Decimal(str(value))
    return (d / step).to_integral_value(rounding=ROUND_DOWN) * step


class BinanceClient:
    def __init__(self) -> None:
        s = get_settings()
        self.testnet = s.binance_testnet
        self._key = s.binance_api_key
        self._secret = s.binance_api_secret
        self.client: Optional[AsyncClient] = None
        self.spot_filters: dict[str, SymbolFilters] = {}
        self.perp_filters: dict[str, SymbolFilters] = {}
        self.time_offset_ms: int = 0
        # Tracks which perp symbols have had isolated/leverage set this run
        # so the Executor stops re-issuing them on every order.
        self.perp_setup_done: set[str] = set()
        # Conservative limits — refined from response headers at runtime.
        self.rest_limiter = AsyncLimiter(max_rate=1200, time_period=60)  # weight
        self.order_limiter = AsyncLimiter(max_rate=50, time_period=10)

    async def start(self) -> None:
        self.client = await AsyncClient.create(self._key, self._secret, testnet=self.testnet)
        # Time sync — clock skew is the #1 reason orders fail with -1021.
        # `time.time()` returns wall-clock seconds since epoch — must match
        # Binance's `serverTime` (wall-clock ms). Previous code used the
        # event loop's monotonic clock, which produced garbage offsets.
        try:
            srv = await self.client.get_server_time()
            local_ms = int(time.time() * 1000)
            self.time_offset_ms = int(srv["serverTime"]) - local_ms
            log.info("binance.time_offset_ms", offset=self.time_offset_ms, testnet=self.testnet)
            if abs(self.time_offset_ms) > 1000:
                log.warning("binance.large_clock_skew_ms", offset=self.time_offset_ms)
        except Exception as e:
            self.time_offset_ms = 0
            log.warning("binance.time_sync_failed", err=str(e))
        await self._load_filters()

    async def close(self) -> None:
        if self.client:
            await self.client.close_connection()

    # ----- Filters / quantization -----
    async def _load_filters(self) -> None:
        assert self.client is not None
        async with self.rest_limiter:
            info = await self.client.get_exchange_info()
        for s in info["symbols"]:
            f = {filt["filterType"]: filt for filt in s["filters"]}
            self.spot_filters[s["symbol"]] = SymbolFilters(
                symbol=s["symbol"],
                tick_size=Decimal(f["PRICE_FILTER"]["tickSize"]),
                step_size=Decimal(f["LOT_SIZE"]["stepSize"]),
                min_notional=Decimal(f.get("NOTIONAL", {}).get("minNotional",
                                     f.get("MIN_NOTIONAL", {}).get("minNotional", "0"))),
                base_asset=s["baseAsset"],
                quote_asset=s["quoteAsset"],
            )
        try:
            async with self.rest_limiter:
                finfo = await self.client.futures_exchange_info()
            for s in finfo["symbols"]:
                if s.get("contractType") != "PERPETUAL":
                    continue
                f = {filt["filterType"]: filt for filt in s["filters"]}
                min_notional = Decimal("0")
                if "MIN_NOTIONAL" in f:
                    min_notional = Decimal(f["MIN_NOTIONAL"].get("notional", "0"))
                self.perp_filters[s["symbol"]] = SymbolFilters(
                    symbol=s["symbol"],
                    tick_size=Decimal(f["PRICE_FILTER"]["tickSize"]),
                    step_size=Decimal(f["LOT_SIZE"]["stepSize"]),
                    min_notional=min_notional,
                    base_asset=s["baseAsset"],
                    quote_asset=s["quoteAsset"],
                )
        except BinanceAPIException as e:
            log.warning("binance.futures_info_failed", err=str(e))

    def quantize(self, symbol: str, qty: float, price: float, market: str
                 ) -> tuple[Decimal, Decimal]:
        table = self.spot_filters if market == "spot" else self.perp_filters
        f = table.get(symbol)
        if not f:
            raise ValueError(f"unknown symbol {symbol} on {market}")
        q = _floor_to_step(qty, f.step_size)
        p = _floor_to_step(price, f.tick_size)
        return q, p

    def passes_min_notional(self, symbol: str, qty: Decimal, price: Decimal, market: str
                            ) -> bool:
        table = self.spot_filters if market == "spot" else self.perp_filters
        f = table.get(symbol)
        if not f:
            return False
        return (qty * price) >= f.min_notional

    # ----- WebSocket streams -----
    async def stream_klines(self, symbols: list[str], interval: str
                            ) -> AsyncIterator[dict]:
        """Resilient kline stream — auto-reconnects on disconnect with
        exponential backoff. Yields raw kline `data` payloads. Consumers
        should re-warm indicators on a "stream.reconnect" sentinel:
            {"e": "stream.reconnect", "interval": interval}
        emitted before resumption. Loops forever; cancel the task to stop."""
        assert self.client is not None
        backoff_s = 1.0
        while True:
            try:
                bm = BinanceSocketManager(self.client)
                streams = [f"{s.lower()}@kline_{interval}" for s in symbols]
                socket = bm.multiplex_socket(streams)
                async with socket as stream:
                    backoff_s = 1.0  # reset on successful (re)connect
                    while True:
                        msg = await stream.recv()
                        data = (msg or {}).get("data") or {}
                        if data.get("e") == "kline":
                            yield data
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("ws.kline.disconnect", interval=interval, err=str(e),
                            backoff_s=backoff_s)
                yield {"e": "stream.reconnect", "interval": interval}
                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 30.0)

    async def stream_agg_trades(self, symbols: list[str]) -> AsyncIterator[dict]:
        assert self.client is not None
        bm = BinanceSocketManager(self.client)
        streams = [f"{s.lower()}@aggTrade" for s in symbols]
        socket = bm.multiplex_socket(streams)
        async with socket as stream:
            while True:
                msg = await stream.recv()
                if msg.get("data", {}).get("e") == "aggTrade":
                    yield msg["data"]

    async def stream_futures_user(self) -> AsyncIterator[dict]:
        assert self.client is not None
        bm = BinanceSocketManager(self.client)
        async with bm.futures_user_socket() as stream:
            while True:
                msg = await stream.recv()
                yield msg

    # ----- REST: market data warmup -----
    async def fetch_klines(self, symbol: str, interval: str, limit: int = 500,
                           market: str = "spot") -> list[list]:
        assert self.client is not None
        async with self.rest_limiter:
            if market == "spot":
                return await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            return await self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)

    # ----- REST: account state -----
    async def spot_balances(self) -> dict[str, float]:
        assert self.client is not None
        async with self.rest_limiter:
            acct = await self.client.get_account()
        return {b["asset"]: float(b["free"]) for b in acct["balances"] if float(b["free"]) > 0}

    async def futures_positions(self) -> list[dict]:
        assert self.client is not None
        async with self.rest_limiter:
            return await self.client.futures_position_information()

    async def futures_account(self) -> dict:
        assert self.client is not None
        async with self.rest_limiter:
            return await self.client.futures_account()

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        assert self.client is not None
        async with self.rest_limiter:
            await self.client.futures_change_leverage(symbol=symbol, leverage=leverage)

    async def set_isolated(self, symbol: str) -> None:
        assert self.client is not None
        try:
            async with self.rest_limiter:
                await self.client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
        except BinanceAPIException as e:
            if e.code != -4046:  # already in that mode
                raise

    # ----- Margin (spot borrow/repay) -----
    async def margin_buy_to_close_short(self, symbol: str, qty: Decimal,
                                         client_order_id: str) -> dict:
        """Buy back borrowed inventory on the margin account, then auto-repay
        the loan. `sideEffectType=AUTO_REPAY` does both atomically."""
        assert self.client is not None
        async with self.order_limiter, self.rest_limiter:
            return await self.client.create_margin_order(
                symbol=symbol, side="BUY", type="MARKET",
                quantity=str(qty), newClientOrderId=client_order_id,
                sideEffectType="AUTO_REPAY",
                isIsolated="FALSE",
            )

    async def margin_sell_with_borrow(self, symbol: str, qty: Decimal,
                                       client_order_id: str) -> dict:
        """Borrow + sell in one shot via `sideEffectType=MARGIN_BUY`. Used for
        opening the short-spot leg of a negative-funding pair trade."""
        assert self.client is not None
        async with self.order_limiter, self.rest_limiter:
            return await self.client.create_margin_order(
                symbol=symbol, side="SELL", type="MARKET",
                quantity=str(qty), newClientOrderId=client_order_id,
                sideEffectType="MARGIN_BUY",
                isIsolated="FALSE",
            )

    async def ensure_perp_setup(self, symbol: str, leverage: int) -> None:
        """Idempotent per-process: set isolated + leverage once per symbol.
        Safe to call on startup for all symbols you intend to trade."""
        if symbol in self.perp_setup_done:
            return
        try:
            await self.set_isolated(symbol)
        except Exception as e:
            log.warning("perp_setup.isolated_failed", symbol=symbol, err=str(e))
        try:
            await self.set_leverage(symbol, max(1, leverage))
        except Exception as e:
            log.warning("perp_setup.leverage_failed", symbol=symbol, err=str(e))
        self.perp_setup_done.add(symbol)

    # ----- REST: order placement -----
    async def place_spot_market(self, symbol: str, side: str, qty: Decimal,
                                client_order_id: str) -> dict:
        assert self.client is not None
        async with self.order_limiter, self.rest_limiter:
            return await self.client.create_order(
                symbol=symbol, side=side, type="MARKET",
                quantity=str(qty), newClientOrderId=client_order_id,
            )

    async def place_perp_market(self, symbol: str, side: str, qty: Decimal,
                                client_order_id: str, reduce_only: bool = False) -> dict:
        assert self.client is not None
        params: dict[str, Any] = {
            "symbol": symbol, "side": side, "type": "MARKET",
            "quantity": str(qty), "newClientOrderId": client_order_id,
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        async with self.order_limiter, self.rest_limiter:
            return await self.client.futures_create_order(**params)

    async def place_perp_stop_market(self, symbol: str, side: str, stop_price: Decimal,
                                     qty: Decimal, client_order_id: str) -> dict:
        assert self.client is not None
        async with self.order_limiter, self.rest_limiter:
            return await self.client.futures_create_order(
                symbol=symbol, side=side, type="STOP_MARKET",
                stopPrice=str(stop_price), quantity=str(qty),
                reduceOnly="true", newClientOrderId=client_order_id,
                workingType="MARK_PRICE",
            )

    async def place_perp_take_profit(self, symbol: str, side: str, stop_price: Decimal,
                                     qty: Decimal, client_order_id: str) -> dict:
        assert self.client is not None
        async with self.order_limiter, self.rest_limiter:
            return await self.client.futures_create_order(
                symbol=symbol, side=side, type="TAKE_PROFIT_MARKET",
                stopPrice=str(stop_price), quantity=str(qty),
                reduceOnly="true", newClientOrderId=client_order_id,
                workingType="MARK_PRICE",
            )

    # ----- REST: misc -----
    async def funding_rate(self, symbol: str) -> Optional[float]:
        assert self.client is not None
        try:
            async with self.rest_limiter:
                px = await self.client.futures_mark_price(symbol=symbol)
            return float(px.get("lastFundingRate", 0.0))
        except Exception:
            return None
