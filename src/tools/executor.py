"""Order execution.

Single chokepoint between approved proposals and Binance. Responsibilities:
- Pre-flight: re-validate symbol against `exchangeInfo`, quantize qty/price,
  check minNotional, set margin/leverage for perps if needed.
- Place entry order with deterministic clientOrderId derived from proposal_id.
- For perps: place reduceOnly stop-market and take-profit orders after entry.
- Persist a Fill row on success.
- Idempotent: a duplicate submit of the same proposal_id is a no-op.

The orchestrator only calls `execute(proposal)` here. All state-machine
transitions live in the orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import structlog

from src.models.types import Proposal, client_order_id
from src.services.storage import Storage
from src.tools.binance_client import BinanceClient

log = structlog.get_logger(__name__)


@dataclass
class ExecutionResult:
    ok: bool
    fill_price: Optional[float] = None
    fill_qty: Optional[float] = None
    entry_order: Optional[dict] = None
    stop_order: Optional[dict] = None
    tp_order: Optional[dict] = None
    error: Optional[str] = None


class Executor:
    def __init__(self, binance: BinanceClient, storage: Storage,
                 paper: bool = False) -> None:
        self.binance = binance
        self.storage = storage
        self.paper = paper
        self._submitted: set[str] = set()  # in-memory idempotency

    async def execute(self, p: Proposal) -> ExecutionResult:
        if p.id in self._submitted:
            log.info("executor.dup_skip", proposal_id=p.id)
            return ExecutionResult(ok=True)
        self._submitted.add(p.id)

        # Paper mode: simulate fill at signal entry with configurable entry
        # slippage (taker market order assumption) + fee. The exit side is
        # handled by PositionManager, which applies its own stop/TP slippage.
        if self.paper:
            from src.config.settings import get_settings as _gs
            s = _gs()
            side = "BUY" if p.signal.side == "long" else "SELL"
            slip_bps = s.slippage_bps
            entry_slip = p.signal.entry * (slip_bps / 10_000)
            fill_px = p.signal.entry + entry_slip if side == "BUY" else p.signal.entry - entry_slip
            fee_bps = s.perps_taker_fee_bps if p.market == "perps" else s.spot_taker_fee_bps
            fee = (p.qty * fill_px) * (fee_bps / 10_000)
            await self.storage.save_fill(
                proposal_id=p.id, symbol=p.signal.symbol, side=side,
                qty=p.qty, price=fill_px, signal_price=p.signal.entry, fee=fee,
                raw={"paper": True, "slip_bps": slip_bps},
            )
            return ExecutionResult(ok=True, fill_price=fill_px, fill_qty=p.qty)

        # --- Live ---
        sym = p.signal.symbol
        side = "BUY" if p.signal.side == "long" else "SELL"
        close_side = "SELL" if side == "BUY" else "BUY"
        coid = client_order_id(p.id)

        try:
            qty_q, _ = self.binance.quantize(sym, p.qty, p.signal.entry, p.market)
            entry_px_q = Decimal(str(p.signal.entry))
            if qty_q <= 0 or not self.binance.passes_min_notional(sym, qty_q, entry_px_q, p.market):
                return ExecutionResult(ok=False, error="min notional / qty after quantization")
        except Exception as e:
            return ExecutionResult(ok=False, error=f"quantize: {e}")

        try:
            if p.market == "spot":
                entry = await self.binance.place_spot_market(sym, side, qty_q, coid)
                price = float(entry.get("fills", [{}])[0].get("price", p.signal.entry)) \
                    if entry.get("fills") else p.signal.entry
                fee = sum(float(f.get("commission", 0)) for f in entry.get("fills", []))
                await self.storage.save_fill(p.id, sym, side, float(qty_q), price, fee, entry,
                                              signal_price=p.signal.entry)
                return ExecutionResult(ok=True, fill_price=price, fill_qty=float(qty_q),
                                       entry_order=entry)

            # Perps: ensure leverage/isolated for THIS symbol before ordering.
            # dfunding trades a dynamic universe, so startup setup for the static
            # symbol_list is not enough — without an explicit low leverage the
            # micro-alts reject the order with -2027 (max position at leverage).
            await self.binance.ensure_perp_setup(sym, getattr(p, "leverage", 1) or 1)
            entry = await self.binance.place_perp_market(sym, side, qty_q, coid)

            # Quantize the stop/TP PRICES to the symbol's tick size — the second
            # return of quantize() is the tick-floored price. Passing the raw
            # price rejects the bracket with -1111 (precision) on tight-tick
            # micro-alts, leaving the position with no exchange stop/TP.
            _, stop_px = self.binance.quantize(sym, 0.0, p.signal.stop, p.market)
            _, tp_px = self.binance.quantize(sym, 0.0, p.signal.take_profit, p.market)
            try:
                stop_order = await self.binance.place_perp_stop_market(
                    sym, close_side, stop_px, qty_q, f"{coid}_s",
                )
            except Exception as e:
                log.warning("executor.stop_failed", err=str(e))
                stop_order = {"error": str(e)}
            try:
                tp_order = await self.binance.place_perp_take_profit(
                    sym, close_side, tp_px, qty_q, f"{coid}_t",
                )
            except Exception as e:
                log.warning("executor.tp_failed", err=str(e))
                tp_order = {"error": str(e)}

            await self.storage.save_fill(
                p.id, sym, side, float(qty_q), p.signal.entry, 0.0, entry,
                signal_price=p.signal.entry,
            )
            return ExecutionResult(ok=True, fill_price=p.signal.entry,
                                   fill_qty=float(qty_q), entry_order=entry,
                                   stop_order=stop_order, tp_order=tp_order)
        except Exception as e:
            log.exception("executor.error", proposal_id=p.id)
            self._submitted.discard(p.id)  # allow retry on transient failure
            return ExecutionResult(ok=False, error=str(e))
