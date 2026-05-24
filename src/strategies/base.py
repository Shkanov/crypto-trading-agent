"""Strategy abstraction — lets multiple alpha sources coexist on one orchestrator.

Each Strategy is a long-running async component that consumes market data
(klines, funding rates, news, etc.) and emits proposals through the
`StrategyContext` it receives at startup. The orchestrator owns:
- risk gate (every proposal is evaluated)
- approval flow (auto / Telegram / 2FA)
- execution (idempotent client_order_id)
- position lifecycle (PositionManager)

Strategies own:
- their own entry/exit logic
- their own state (with hooks to persist via `ctx.storage`)
- their own cadence / event subscriptions

This lets us run e.g. IndicatorConfluence + FundingHarvest side-by-side; the
risk gate sees both as competing claims on the same equity.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol

from src.config.settings import Settings
from src.models.types import (
    IndicatorSnapshot,
    Kline,
    PairProposal,
    Signal,
    Trade,
)
from src.services.event_bus import EventBus
from src.services.storage import Storage
from src.tools.binance_client import BinanceClient
from src.tools.indicators import IndicatorEngine
from src.tools.pair_executor import PairExecutor


class StrategyContext(Protocol):
    """Hooks the orchestrator exposes to strategies.

    Strategies should treat this as their entire view of the system — they
    must not import the orchestrator directly (avoid cycles, keep testable).
    """
    settings: Settings
    storage: Storage
    bus: EventBus
    indicators: IndicatorEngine
    binance: BinanceClient
    last_price: dict[str, float]
    # Optional: pair-trade strategies use this to close legs at correct
    # per-market prices (spot mid vs perp mark).
    pair_executor: Optional[PairExecutor]

    async def propose(self, signal: Signal, market: str, leverage: int,
                       strategy_name: str) -> None:
        ...

    async def propose_pair(self, pair: PairProposal) -> None:
        ...

    async def close_trade(self, trade_id: str, reason: str) -> None:
        ...

    def open_trades(self, strategy_name: Optional[str] = None) -> list[Trade]:
        ...

    def equity_available_usd(self, strategy_name: Optional[str] = None) -> float:
        ...


class Strategy(ABC):
    """Base class for a strategy. Subclass and implement at minimum `start`
    and `name`."""
    name: str = "unnamed"

    @abstractmethod
    async def start(self, ctx: StrategyContext) -> None:
        """Begin reacting to events. Should be non-blocking — typically spins
        up its own task(s) on the asyncio loop and returns."""
        raise NotImplementedError

    async def stop(self) -> None:
        """Cancel any owned tasks and release resources. Default is no-op."""
        return None

    # Optional hooks — strategies override only those they care about.

    async def on_bar(self, k: Kline, snap: IndicatorSnapshot) -> None:
        """Called on each CLOSED kline. Default no-op."""
        return None

    async def on_trade_closed(self, trade: Trade) -> None:
        """Called when a trade owned by this strategy closes. Default no-op."""
        return None
