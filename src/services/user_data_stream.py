"""Futures user-data-stream consumer — the live source of truth for funding
payments, fills, and position state.

Binance futures user-data events relevant to us:

- ``ACCOUNT_UPDATE`` (``e=ACCOUNT_UPDATE``): balance + position deltas. We use
  it for reconciliation drift detection during a run.
- ``ORDER_TRADE_UPDATE`` (``e=ORDER_TRADE_UPDATE``): fill events with REAL
  prices and fees (REST often returns pre-fill prices). We use it to update
  Trade entry_price / fee_total_usd from actual execution.
- ``FUNDING_FEE`` events are delivered as an income-update inside
  ``ACCOUNT_UPDATE.B`` (balance) or via the ``income`` REST. Binance also
  emits a dedicated payload that we recognize by ``i.E=FUNDING_FEE`` shape.
  We use it to credit Trade.funding_accrued_usd on the live exchange's
  authoritative payment, replacing the paper accrual loop.

Open trades are matched to incoming events by (symbol, market). When multiple
open Trades exist on the same (symbol, market), we credit/debit the OLDEST
open one — most-recently-opened pairs that haven't crossed an 8h boundary
shouldn't be receiving funding yet.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

import structlog

from src.services.storage import Storage
from src.tools.binance_client import BinanceClient

log = structlog.get_logger(__name__)


class UserDataStream:
    """Runs as a long-lived task; emits parsed events to registered handlers.

    Resilience: the underlying `binance.futures_user_socket` raises on
    disconnect; we restart with exponential backoff. listenKey keepalive
    is handled inside python-binance's manager (PUTs every ~30 min)."""

    def __init__(
        self,
        binance: BinanceClient,
        storage: Storage,
        on_fill: Optional[Callable[[dict], Awaitable[None]]] = None,
        on_account_update: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> None:
        self.binance = binance
        self.storage = storage
        self.on_fill = on_fill
        self.on_account_update = on_account_update
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async for msg in self.binance.stream_futures_user():
                    backoff = 1.0
                    await self._handle(msg)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("user_data_stream.disconnect", err=str(e), backoff_s=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _handle(self, msg: dict) -> None:
        e = msg.get("e")
        if e == "ACCOUNT_UPDATE":
            await self._on_account_update(msg)
        elif e == "ORDER_TRADE_UPDATE":
            await self._on_order_trade_update(msg)
        elif e == "MARGIN_CALL":
            log.critical("user_data.MARGIN_CALL", payload=msg)
        elif e == "listenKeyExpired":
            log.warning("user_data.listenKeyExpired")
        # FUNDING_FEE events also arrive embedded in account updates — see ACCOUNT_UPDATE handler.

    async def _on_account_update(self, msg: dict) -> None:
        """ACCOUNT_UPDATE arrives on order fills, funding, transfers, etc.

        We DO NOT credit funding from here — the WS payload combines multi-
        symbol balance deltas without per-symbol breakdown, which would
        over-attribute on accounts with multiple open positions (R1 bug).
        Funding income is now polled from `/fapi/v1/income?incomeType=FUNDING_FEE`
        by FundingIncomePoller — authoritative per-symbol attribution.

        We still surface the raw event to the orchestrator's optional handler
        so it can update position state or audit balances."""
        if self.on_account_update:
            try:
                await self.on_account_update(msg)
            except Exception:
                log.exception("user_data.on_account_update.failed")

    async def _on_order_trade_update(self, msg: dict) -> None:
        """`o` field contains the order/trade detail. Refines our Trade record
        with REAL fill price + fee (REST response is often pre-fill)."""
        o = msg.get("o") or {}
        status = o.get("X")  # FILLED, PARTIALLY_FILLED, NEW, CANCELED, EXPIRED, ...
        if status not in ("FILLED", "PARTIALLY_FILLED"):
            if self.on_fill:
                try:
                    await self.on_fill({"status": status, "raw": msg})
                except Exception:
                    log.exception("user_data.on_fill.passthrough.failed")
            return
        coid = o.get("c", "")
        # Our deterministic client_order_id starts with "cta_" or "pair_".
        if not (coid.startswith("cta_") or coid.startswith("pair_")):
            return
        last_filled_price = float(o.get("L") or o.get("ap") or 0)
        commission = float(o.get("n") or 0)
        # The fill amount this update reports (delta).
        l_qty = float(o.get("l") or 0)
        symbol = o.get("s")
        log.info("user_data.fill", coid=coid, symbol=symbol,
                 price=last_filled_price, qty=l_qty, fee=commission, status=status)
        if self.on_fill:
            try:
                await self.on_fill({
                    "status": status, "coid": coid, "symbol": symbol,
                    "price": last_filled_price, "qty": l_qty,
                    "fee": commission, "raw": msg,
                })
            except Exception:
                log.exception("user_data.on_fill.failed")
