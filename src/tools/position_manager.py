"""Position lifecycle management.

In PAPER mode this module is the source of truth for closes:
  - Watches each kline (high/low/close) on the symbol's trigger timeframe
  - For each open Trade, detects if the bar's range touched the intended
    stop or take-profit
  - Closes with realistic slippage (configurable per-side, larger on stops)
  - Persists the closed Trade, updates orchestrator-visible state

In LIVE mode this module only handles MANUAL closes (panic flatten, anomaly
"flatten" action, time-stop). Bracket orders on the exchange handle stop/TP;
we reconcile via the user data stream (wired separately).

Returns closed Trades via a callback (`on_close`) so the orchestrator can
update its in-memory position list, consecutive-loss counter, etc.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import structlog

from src.config.settings import Settings, get_settings
from src.models.types import Trade, now_ms
from src.services.storage import Storage

log = structlog.get_logger(__name__)


@dataclass
class TickRange:
    """One bar's high/low/close on a symbol. The position manager evaluates
    stop/TP touches with conservative slippage assumptions."""
    symbol: str
    high: float
    low: float
    close: float
    ts_ms: int


def _exit_with_slippage(intended: float, side: str, reason: str,
                        stop_slip_bps: float, tp_slip_bps: float) -> float:
    """Simulate a fill at `intended` with side-appropriate adverse slippage.

    For a stop hit, the fill is WORSE than `intended` by `stop_slip_bps`.
    For a TP hit, the fill is WORSE than `intended` by `tp_slip_bps` (limit
    orders often partial-fill into adverse moves; we approximate)."""
    bps = stop_slip_bps if reason == "stop" else tp_slip_bps
    move = intended * (bps / 10_000)
    # long position closing: SELL. Adverse = lower price. short close: BUY. Adverse = higher.
    if side == "long":
        return intended - move
    return intended + move


class PositionManager:
    """Tracks open Trades; evaluates stop/TP on each new bar in paper mode."""

    def __init__(
        self,
        storage: Storage,
        on_close: Callable[[Trade], Awaitable[None]],
        paper: bool = True,
        settings: Optional[Settings] = None,
        close_executor: Optional[Callable[[Trade], Awaitable[Optional[float]]]] = None,
    ) -> None:
        self.storage = storage
        self.on_close = on_close
        self.paper = paper
        self.s = settings or get_settings()
        # LIVE only: places the real reduceOnly exchange order to close a
        # position and returns the fill price (None on failure). Without it,
        # force_close would only update the DB and leave the position open on
        # the exchange — the phantom-close desync. Paper mode leaves this None.
        self.close_executor = close_executor
        # In-memory mirror of OPEN trades, keyed by trade_id. Rebuilt from
        # storage on rehydrate() so a process restart preserves positions.
        self.open: dict[str, Trade] = {}

    async def rehydrate(self) -> None:
        for t in await self.storage.list_open_trades():
            self.open[t.id] = t
        log.info("position_manager.rehydrated", n=len(self.open))

    async def register(self, trade: Trade) -> None:
        """Called after a successful entry fill. Persists OPEN row + tracks."""
        await self.storage.open_trade(trade)
        self.open[trade.id] = trade

    async def on_bar(self, bar: TickRange) -> None:
        """Evaluate every open trade on this symbol against the bar's range.

        Conservatism: if BOTH stop and TP would be hit within the same bar,
        we resolve the STOP (it's the worse outcome for the trader and the
        realistic assumption without intra-bar tick data)."""
        if not self.paper:
            return
        closed: list[tuple[Trade, float, str]] = []
        for tid, t in list(self.open.items()):
            if t.symbol != bar.symbol:
                continue
            hit_stop = False
            hit_tp = False
            if t.side == "long":
                if t.intended_stop is not None and bar.low <= t.intended_stop:
                    hit_stop = True
                if t.intended_tp is not None and bar.high >= t.intended_tp:
                    hit_tp = True
            else:
                if t.intended_stop is not None and bar.high >= t.intended_stop:
                    hit_stop = True
                if t.intended_tp is not None and bar.low <= t.intended_tp:
                    hit_tp = True

            if hit_stop:
                exit_px = _exit_with_slippage(
                    t.intended_stop or t.entry_price, t.side, "stop",
                    self.s.paper_stop_slippage_bps, self.s.paper_tp_slippage_bps,
                )
                closed.append((t, exit_px, "stop"))
            elif hit_tp:
                exit_px = _exit_with_slippage(
                    t.intended_tp or t.entry_price, t.side, "tp",
                    self.s.paper_stop_slippage_bps, self.s.paper_tp_slippage_bps,
                )
                closed.append((t, exit_px, "tp"))

        for trade, exit_px, reason in closed:
            await self._finalize(trade, exit_px, reason)

    async def force_close(self, trade_id: str, exit_price: float, reason: str = "manual") -> Optional[Trade]:
        """Actively close a position. In LIVE mode this places the real
        reduceOnly exchange order FIRST via `close_executor`; the DB row is
        marked closed ONLY on a confirmed fill, so a failed exchange close can
        never leave a phantom-closed (DB-closed / exchange-open) position.
        Paper mode records the simulated close directly."""
        t = self.open.get(trade_id)
        if not t:
            return None
        if not self.paper and self.close_executor is not None:
            try:
                fill = await self.close_executor(t)
            except Exception as e:  # noqa: BLE001
                log.error("position_manager.live_close_raised",
                          trade_id=trade_id, symbol=t.symbol, err=str(e))
                return None
            if fill is None or fill <= 0:
                log.error("position_manager.live_close_failed",
                          trade_id=trade_id, symbol=t.symbol, reason=reason)
                return None   # keep it OPEN in self.open — no phantom close
            exit_price = fill
        return await self._finalize(t, exit_price, reason)

    async def force_close_all(self, last_prices: dict[str, float], reason: str = "flatten"
                              ) -> list[Trade]:
        results: list[Trade] = []
        for tid, t in list(self.open.items()):
            px = last_prices.get(t.symbol, t.entry_price)
            # Route through force_close so LIVE mode places real orders and a
            # failed leg is left OPEN rather than phantom-closed.
            closed = await self.force_close(tid, px, reason)
            if closed:
                results.append(closed)
        return results

    async def _finalize(self, t: Trade, exit_px: float, reason: str) -> Optional[Trade]:
        # Fee model: paper applies taker fee on both legs (worst case for honesty)
        fee_bps = self.s.perps_taker_fee_bps if t.market == "perps" else self.s.spot_taker_fee_bps
        exit_fee = (t.qty * exit_px) * (fee_bps / 10_000)
        slip_bps = None
        if t.side == "long":
            ref = t.intended_stop if reason == "stop" else (t.intended_tp if reason == "tp" else exit_px)
            if ref:
                slip_bps = ((ref - exit_px) / ref) * 10_000
        else:
            ref = t.intended_stop if reason == "stop" else (t.intended_tp if reason == "tp" else exit_px)
            if ref:
                slip_bps = ((exit_px - ref) / ref) * 10_000

        closed = await self.storage.close_trade(
            t.id, exit_price=exit_px, exit_reason=reason,
            fee_total_usd=exit_fee, slippage_bps_exit=slip_bps,
        )
        self.open.pop(t.id, None)
        if closed:
            await self.on_close(closed)
            log.info("position_manager.closed",
                     trade_id=t.id, symbol=t.symbol, side=t.side, reason=reason,
                     entry=t.entry_price, exit=exit_px,
                     pnl_usd=closed.realized_pnl_usd)
        return closed
