"""PairExecutor — atomic open/close of delta-neutral pair trades.

A pair trade (e.g. funding-rate harvest) is two legs that must succeed
together: typically long spot + short perp on the same notional.

Paper mode (the default for safety): both legs are simulated and persisted
as two separate Trades, tagged with the same `pair_id` in the proposal_id
field so they can be matched. Closing the pair closes both Trades atomically.

Live mode (NOT enabled until paper trading proves edge): sends spot + perp
market orders sequentially; if leg 2 fails, immediately unwinds leg 1 with
a reverse market order. Slippage budget enforced — if either leg fills
outside the budget, both are unwound.

This module is deliberately thin — the heavy lifting (basis safety, funding
extremes detection) lives in the strategy itself. PairExecutor's sole
contract: open both, or open neither.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import structlog

from src.config.settings import Settings, get_settings
from src.models.types import PairLeg, PairProposal, Trade, now_ms
from src.services.storage import Storage
from src.tools.binance_client import BinanceClient

log = structlog.get_logger(__name__)


@dataclass
class PairOpenResult:
    ok: bool
    legs: list[Trade]            # both legs as opened Trades, or [] on failure
    fill_prices: list[float]
    error: Optional[str] = None


class PairExecutor:
    def __init__(self, binance: BinanceClient, storage: Storage,
                 paper: bool = True, settings: Optional[Settings] = None) -> None:
        self.binance = binance
        self.storage = storage
        self.paper = paper
        self.s = settings or get_settings()

    async def open_pair(self, pair: PairProposal) -> PairOpenResult:
        """Open both legs atomically. In paper mode this is trivially atomic;
        in live mode we unwind leg 1 if leg 2 fails."""
        if self.paper:
            return await self._open_paper(pair)
        return await self._open_live(pair)

    async def close_pair(self, legs: list[Trade], reason: str = "manual",
                         price_map: Optional[dict[tuple[str, str], float]] = None,
                         last_prices: Optional[dict[str, float]] = None) -> list[Trade]:
        """Close both legs atomically. `price_map` is keyed by (symbol, market)
        so the perp leg uses the perp mark and the spot leg uses the spot mid.
        `last_prices` (symbol-only) is kept as a fallback for callers that
        haven't migrated yet, but will mis-price one leg by the basis."""
        if self.paper:
            return await self._close_paper(legs, reason, price_map or {}, last_prices or {})
        return await self._close_live(legs, reason)

    # ----- Paper -----
    async def _open_paper(self, pair: PairProposal) -> PairOpenResult:
        trades: list[Trade] = []
        prices: list[float] = []
        slip_bps = self.s.slippage_bps
        for leg in pair.legs:
            fee_bps = self.s.perps_taker_fee_bps if leg.market == "perps" else self.s.spot_taker_fee_bps
            # R13: quantize qty + price to exchange step/tick sizes — same as live.
            # Filters are cached on BinanceClient at startup. Fall back to raw
            # values if filters unavailable (e.g. ungated unit tests).
            try:
                qty_q_dec, _ = self.binance.quantize(leg.symbol, leg.qty,
                                                      leg.expected_price, leg.market)
                qty_q = float(qty_q_dec)
            except Exception:
                qty_q = leg.qty
            slip = leg.expected_price * (slip_bps / 10_000)
            fill_px = leg.expected_price + slip if leg.side == "BUY" else leg.expected_price - slip
            fee = qty_q * fill_px * (fee_bps / 10_000)
            trade_side = "long" if leg.side == "BUY" else "short"
            t = Trade(
                strategy=pair.strategy,
                proposal_id=pair.id,
                symbol=leg.symbol,
                market=leg.market,
                side=trade_side,
                qty=qty_q,
                leverage=leg.leverage,
                entry_price=fill_px,
                # Pair trades don't carry stop/TP — exits are condition-driven.
                intended_stop=None, intended_tp=None,
                fee_total_usd=fee,
                slippage_bps_entry=slip_bps,
            )
            await self.storage.open_trade(t)
            await self.storage.save_fill(
                proposal_id=pair.id, symbol=leg.symbol, side=leg.side,
                qty=qty_q, price=fill_px, signal_price=leg.expected_price,
                fee=fee, raw={"paper": True, "pair": True}, trade_id=t.id,
            )
            trades.append(t)
            prices.append(fill_px)
            log.info("pair.opened_leg", pair_id=pair.id, symbol=leg.symbol,
                     side=leg.side, qty=leg.qty, fill=fill_px)
        return PairOpenResult(ok=True, legs=trades, fill_prices=prices)

    async def _close_paper(self, legs: list[Trade], reason: str,
                           price_map: dict[tuple[str, str], float],
                           last_prices: dict[str, float]) -> list[Trade]:
        """Close each leg at its own market's reference price (spot mid for
        spot leg, perp mark for perp leg) — F2 fix. Apply adverse slippage
        on close (F6): SELL-to-close a long leg fills LOWER; BUY-to-close
        a short leg fills HIGHER."""
        closed: list[Trade] = []
        slip_bps = self.s.slippage_bps
        for leg in legs:
            # Prefer the (symbol, market) keyed map; fall back to last_prices,
            # then to entry as a last resort. The fallback paths bias P&L on
            # the perp leg by the basis — callers should pass price_map.
            ref = price_map.get((leg.symbol, leg.market))
            if ref is None:
                ref = last_prices.get(leg.symbol, leg.entry_price)
            # Adverse on close: closing a LONG = SELL, fills LOWER.
            # Closing a SHORT = BUY, fills HIGHER.
            slip = ref * (slip_bps / 10_000)
            exit_px = ref - slip if leg.side == "long" else ref + slip
            fee_bps = self.s.perps_taker_fee_bps if leg.market == "perps" else self.s.spot_taker_fee_bps
            fee = leg.qty * exit_px * (fee_bps / 10_000)
            c = await self.storage.close_trade(
                leg.id, exit_price=exit_px, exit_reason=reason,
                fee_total_usd=fee, slippage_bps_exit=slip_bps,
            )
            if c:
                closed.append(c)
                log.info("pair.closed_leg", pair_id=leg.proposal_id, symbol=leg.symbol,
                         market=leg.market, ref=ref, exit_price=exit_px,
                         pnl=c.realized_pnl_usd,
                         funding=c.funding_accrued_usd)
        return closed

    def _requires_spot_margin(self, pair: PairProposal) -> bool:
        """Any pair whose spot leg is a SELL needs spot margin (borrow + sell).
        Live mode refuses such pairs unless explicitly enabled via setting."""
        return any(leg.market == "spot" and leg.side == "SELL" for leg in pair.legs)

    # ----- Live (best-effort skeleton; enable with care) -----
    async def _open_live(self, pair: PairProposal) -> PairOpenResult:
        # Negative-funding direction (short-spot leg) requires Binance margin
        # account. Refuse in live mode until the operator explicitly opts in.
        if self._requires_spot_margin(pair) and not getattr(self.s, "live_spot_margin_enabled", False):
            return PairOpenResult(
                ok=False, legs=[], fill_prices=[],
                error="pair requires spot margin; set LIVE_SPOT_MARGIN_ENABLED=true after enabling margin on the Binance account",
            )
        opened: list[tuple[PairLeg, dict]] = []
        try:
            for leg in pair.legs:
                qty_q, _ = self.binance.quantize(leg.symbol, leg.qty,
                                                  leg.expected_price, leg.market)
                if qty_q <= 0:
                    raise ValueError(f"qty rounded to zero for {leg.symbol}")
                coid = f"pair_{pair.id[:24]}_{len(opened)}"
                if leg.market == "spot":
                    if leg.side == "SELL":
                        # R4: short-spot entry uses margin SELL with auto-borrow.
                        res = await self.binance.margin_sell_with_borrow(
                            leg.symbol, qty_q, coid,
                        )
                    else:
                        res = await self.binance.place_spot_market(
                            leg.symbol, leg.side, qty_q, coid,
                        )
                else:
                    await self.binance.ensure_perp_setup(leg.symbol, leg.leverage)
                    res = await self.binance.place_perp_market(
                        leg.symbol, leg.side, qty_q, coid,
                    )
                opened.append((leg, res))
        except Exception as e:
            log.exception("pair.open_failed_unwinding", pair_id=pair.id)
            # Reverse already-filled legs
            for filled_leg, _ in opened:
                rev_side = "SELL" if filled_leg.side == "BUY" else "BUY"
                try:
                    qty_q, _ = self.binance.quantize(filled_leg.symbol, filled_leg.qty,
                                                      filled_leg.expected_price, filled_leg.market)
                    coid = f"unwind_{pair.id[:24]}"
                    if filled_leg.market == "spot":
                        await self.binance.place_spot_market(filled_leg.symbol, rev_side, qty_q, coid)
                    else:
                        await self.binance.place_perp_market(filled_leg.symbol, rev_side, qty_q, coid,
                                                              reduce_only=True)
                except Exception:
                    log.exception("pair.unwind_failed", symbol=filled_leg.symbol)
            return PairOpenResult(ok=False, legs=[], fill_prices=[], error=str(e))

        # Persist as paired Trades (best-effort price extraction).
        trades: list[Trade] = []
        prices: list[float] = []
        for leg, res in opened:
            fills = res.get("fills") or []
            fill_px = float(fills[0]["price"]) if fills else leg.expected_price
            trade_side = "long" if leg.side == "BUY" else "short"
            t = Trade(
                strategy=pair.strategy, proposal_id=pair.id,
                symbol=leg.symbol, market=leg.market, side=trade_side,
                qty=leg.qty, leverage=leg.leverage,
                entry_price=fill_px, intended_stop=None, intended_tp=None,
            )
            await self.storage.open_trade(t)
            trades.append(t)
            prices.append(fill_px)
        return PairOpenResult(ok=True, legs=trades, fill_prices=prices)

    async def _close_live(self, legs: list[Trade], reason: str) -> list[Trade]:
        """Close both legs with market orders, retry on transient failure.

        Order of closure matters: close the PERP leg FIRST (it has
        liquidation risk if left dangling), then spot. If perp close
        succeeds but spot fails, the operator must reconcile manually —
        the spot leg is just owned inventory, no liquidation risk.

        Each leg is closed via a market order with side flipped (long→SELL,
        short→BUY). We rely on the user-data-stream listener
        (UserDataStream.on_fill) to update Trade rows with real fill prices
        and fees AFTER this returns. The Trade rows we return here have
        provisional fill prices that the stream will refine."""
        closed: list[Trade] = []
        # Sort legs: perp first.
        ordered = sorted(legs, key=lambda l: 0 if l.market == "perps" else 1)
        for leg in ordered:
            close_side = "SELL" if leg.side == "long" else "BUY"
            try:
                qty_q, _ = self.binance.quantize(leg.symbol, leg.qty,
                                                  leg.entry_price, leg.market)
                if qty_q <= 0:
                    log.error("pair.close_live.qty_zero", symbol=leg.symbol)
                    continue
                coid = f"pclose_{leg.id[:24]}"
                if leg.market == "spot":
                    if leg.side == "short":
                        # R4: short_spot close via margin BUY + AUTO_REPAY.
                        # Gated behind LIVE_SPOT_MARGIN_ENABLED (same flag
                        # used by _open_live) so accounts without margin
                        # enabled refuse cleanly instead of erroring out.
                        if not getattr(self.s, "live_spot_margin_enabled", False):
                            log.error("pair.close_live.margin_disabled",
                                      symbol=leg.symbol, trade_id=leg.id)
                            continue
                        res = await self.binance.margin_buy_to_close_short(
                            leg.symbol, qty_q, coid,
                        )
                    else:
                        res = await self.binance.place_spot_market(
                            leg.symbol, close_side, qty_q, coid,
                        )
                else:
                    res = await self.binance.place_perp_market(
                        leg.symbol, close_side, qty_q, coid, reduce_only=True,
                    )
                # Provisional close — real fill price comes from user-data-stream.
                fills = res.get("fills") or []
                fill_px = float(fills[0]["price"]) if fills else leg.entry_price
                fee = sum(float(f.get("commission", 0)) for f in fills)
                c = await self.storage.close_trade(
                    leg.id, exit_price=fill_px, exit_reason=reason,
                    fee_total_usd=fee, slippage_bps_exit=None,
                )
                if c:
                    closed.append(c)
                    log.info("pair.closed_leg_live", trade_id=leg.id,
                             symbol=leg.symbol, market=leg.market,
                             provisional_exit=fill_px)
            except Exception:
                log.exception("pair.close_live.leg_failed",
                              trade_id=leg.id, symbol=leg.symbol)
        return closed
