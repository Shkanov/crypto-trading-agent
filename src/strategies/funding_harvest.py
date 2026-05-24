"""Funding-rate harvesting strategy (delta-neutral).

Real, persistent, capacity-limited retail edge. Mechanics:

When funding rate on a perp is sufficiently positive (longs paying), we:
  long_spot  + short_perp   on equal notional → collect funding on the perp
When funding is sufficiently negative (shorts paying), we:
  short_spot + long_perp    → spot leg via margin (skipped if no spot-margin enabled;
                              for now only the long-funding direction is supported,
                              which is by far the more common regime on majors)

Entry conditions (ALL must hold):
  1. Current funding > `entry_threshold_bps` (default 10 bps per 8h)
  2. 21-period avg funding > `entry_avg_threshold_bps` (default 5 bps) — proves the
     regime, not just a one-print spike
  3. Basis monitor says safe (|basis| < entry_block_bps)
  4. No existing pair on this symbol for THIS strategy
  5. We have spot inventory (or capital) to deploy

Exit conditions (ANY triggers close):
  1. Funding crosses below `exit_threshold_bps` (default 2 bps) — regime over
  2. Basis blowup > exit_alert_bps (basis risk dominates the yield)
  3. Manual flatten / anomaly action

The strategy runs its own poll loop (every funding_check_interval_s). It
does NOT generate Proposals for single-leg trades — it uses PairProposal
through the orchestrator's `propose_pair`.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional

import structlog

from src.config.settings import Settings, get_settings
from src.models.types import PairLeg, PairProposal, Trade, now_ms
from src.services.basis_monitor import BasisMonitor
from src.services.funding_monitor import FundingMonitor
from src.strategies.base import Strategy, StrategyContext

log = structlog.get_logger(__name__)


@dataclass
class HarvestParams:
    entry_threshold_bps: float = 10.0          # current funding must exceed
    entry_avg_threshold_bps: float = 5.0       # 21-period mean must exceed (regime)
    exit_threshold_bps: float = 2.0            # close when funding crosses below
    notional_per_pair_usd: float = 100.0       # spot + perp legs each on this notional
    max_concurrent_pairs: int = 2
    poll_interval_s: int = 60
    spot_tf_for_basis: str = "5m"
    # Leg leverage (perp). 2x isolated keeps capital efficient; the
    # price-move stop below is the explicit safety net against liquidation.
    perp_leverage: int = 2
    # Close the pair if perp mark moves > this much adverse from entry.
    # At 2x isolated, liquidation is around +48% adverse. We exit at 30%
    # to leave buffer before the exchange force-closes the short leg
    # (which would leave the spot leg sitting unmatched).
    perp_adverse_move_pct: float = 30.0
    # Enable the short_spot + long_perp direction (requires Binance spot margin).
    # In paper mode this is always simulated; in live mode the executor must
    # be wired against a margin-enabled account.
    allow_negative_direction: bool = True
    # Spot-margin borrow rate (bps/8h) used in paper accrual for short-spot leg.
    # Real Binance margin rates are ~0.5-3 bps/8h on majors.
    spot_borrow_bps_per_8h: float = 1.5


@dataclass
class ActivePair:
    pair_id: str
    symbol: str
    legs: list[Trade] = field(default_factory=list)
    opened_ms: int = 0
    direction: int = 1  # +1 = long_spot/short_perp, -1 = short_spot/long_perp
    # Tracks the last funding event we've already credited (next_funding_ms
    # value at the time of crediting). Prevents double-paying when the
    # accrual loop runs multiple times within an 8h window.
    last_credited_funding_ms: int = 0
    perp_entry_mark: float = 0.0    # for the price-move stop (F4)

    @property
    def short_perp_leg(self) -> Optional[Trade]:
        for leg in self.legs:
            if leg.market == "perps" and leg.side == "short":
                return leg
        return None

    @property
    def perp_leg(self) -> Optional[Trade]:
        for leg in self.legs:
            if leg.market == "perps":
                return leg
        return None

    @property
    def spot_leg(self) -> Optional[Trade]:
        for leg in self.legs:
            if leg.market == "spot":
                return leg
        return None


class FundingHarvestStrategy(Strategy):
    name = "funding_harvest"

    def __init__(self, funding: FundingMonitor, basis: BasisMonitor,
                 params: Optional[HarvestParams] = None,
                 symbols: Optional[list[str]] = None) -> None:
        self.funding = funding
        self.basis = basis
        self.p = params or HarvestParams()
        self.symbols = symbols or []
        self.ctx: Optional[StrategyContext] = None
        self.active: dict[str, ActivePair] = {}   # by symbol
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self, ctx: StrategyContext) -> None:
        self.ctx = ctx
        if not self.symbols:
            self.symbols = list(ctx.settings.symbol_list)
        # Rehydrate any open pairs that survived a restart. R6: pull
        # pair-level metadata (direction, perp_entry_mark, funding watermark)
        # from PairMetaRow so the F4 safety stop and direction-aware logic
        # work correctly post-restart.
        try:
            existing = await ctx.storage.list_open_trades()
            for t in existing:
                if t.strategy != self.name:
                    continue
                ap = self.active.get(t.symbol)
                if ap is None:
                    meta = await ctx.storage.load_pair_meta(t.proposal_id)
                    if meta:
                        ap = ActivePair(
                            pair_id=t.proposal_id, symbol=t.symbol,
                            opened_ms=meta["opened_ms"],
                            direction=meta["direction"],
                            perp_entry_mark=meta["perp_entry_mark"],
                            last_credited_funding_ms=meta["last_credited_funding_ms"],
                        )
                    else:
                        # No metadata row — most likely from an older deploy
                        # before R6. Best effort: infer direction from leg
                        # sides (long_spot+short_perp = +1; short_spot+long_perp = -1).
                        ap = ActivePair(
                            pair_id=t.proposal_id, symbol=t.symbol,
                            opened_ms=t.entry_ts_ms,
                        )
                    self.active[t.symbol] = ap
                ap.legs.append(t)
                # If we had no PairMetaRow, infer direction from leg sides.
                if ap.direction == 1 and len(ap.legs) >= 2:
                    spot = ap.spot_leg
                    if spot and spot.side == "short":
                        ap.direction = -1
                # Fill perp_entry_mark from the perp leg's entry_price if blank.
                if ap.perp_entry_mark == 0.0:
                    perp = ap.perp_leg
                    if perp:
                        ap.perp_entry_mark = perp.entry_price
        except Exception:
            log.exception("funding_harvest.rehydrate_failed")
        self._task = asyncio.create_task(self._loop())
        log.info("funding_harvest.started", symbols=self.symbols,
                 entry_bps=self.p.entry_threshold_bps,
                 exit_bps=self.p.exit_threshold_bps,
                 notional=self.p.notional_per_pair_usd)

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._evaluate_once()
            except Exception:
                log.exception("funding_harvest.loop_error")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.p.poll_interval_s)
            except asyncio.TimeoutError:
                pass

    async def _evaluate_once(self) -> None:
        ctx = self.ctx
        if not ctx:
            return
        # 0. Credit funding on all open pairs (paper mode in-process; live
        #    mode would read from user-data-stream FUNDING_FEE events instead).
        for sym, ap in list(self.active.items()):
            await self._accrue_funding_if_due(ctx, sym, ap)
        # 1. Check exits — pairs we already hold.
        for sym, ap in list(self.active.items()):
            await self._maybe_close(ctx, sym, ap)
        # 2. Look for new entries.
        if len(self.active) >= self.p.max_concurrent_pairs:
            return
        for sym in self.symbols:
            if sym in self.active:
                continue
            if len(self.active) >= self.p.max_concurrent_pairs:
                break
            await self._maybe_open(ctx, sym)

    async def _accrue_funding_if_due(self, ctx: StrategyContext, symbol: str,
                                      ap: ActivePair) -> None:
        """Credit the perp leg of `ap` when a funding event has settled.
        Detect boundary by watching FundingMonitor's `next_funding_ms` advance.

        Sign convention by leg + rate:
          SHORT perp + rate>0  → COLLECT (positive amount on the short leg)
          SHORT perp + rate<0  → PAY     (negative on the short leg)
          LONG  perp + rate>0  → PAY     (negative on the long leg)
          LONG  perp + rate<0  → COLLECT (positive on the long leg)

        Math: `payment_to_long = -qty * mark * rate`. The receiving side
        equals `+qty * mark * rate` for a LONG, opposite for a SHORT.

        Additionally, the short-spot leg (negative-funding direction) accrues
        a borrow cost — modeled here as a flat bps/8h subtracted from the
        spot leg's funding_accrued each period."""
        fp = self.funding.current(symbol)
        if not fp or fp.mark_price <= 0:
            return
        perp_leg = ap.perp_leg
        spot_leg = ap.spot_leg
        if not perp_leg:
            return
        if ap.last_credited_funding_ms == 0:
            # R12: first observation since open (or rehydrate). Set the
            # watermark to the current next_funding_ms so we don't credit
            # funding that paid BEFORE we opened. Side-effect: the most
            # recently settled funding payment that happened DURING agent
            # downtime is skipped — acceptable trade-off vs guessing.
            ap.last_credited_funding_ms = fp.next_funding_ms
            try:
                await ctx.storage.update_pair_funding_watermark(
                    ap.pair_id, fp.next_funding_ms,
                )
            except Exception:
                log.exception("funding_harvest.initial_watermark_persist_failed",
                              pair_id=ap.pair_id)
            return
        if fp.next_funding_ms > ap.last_credited_funding_ms:
            st = self.funding.state.get(symbol)
            settled_rate = fp.rate
            if st and len(st.history) >= 2:
                settled_rate = st.history[-2].rate
            # R11: clamp to ±50 bps and alert if Binance ever emits an
            # anomalous rate (has happened post-halt). A real rate beyond
            # 50 bps in one 8h window indicates a regime/data event the
            # operator should look at — don't silently apply it.
            ABS_CAP = 50.0 / 10_000   # 50 bps = 0.005
            if abs(settled_rate) > ABS_CAP:
                log.critical("funding.rate_anomalous",
                             symbol=symbol, raw_rate_bps=settled_rate * 10_000,
                             clamped_to_bps=(ABS_CAP if settled_rate > 0 else -ABS_CAP) * 10_000)
                settled_rate = ABS_CAP if settled_rate > 0 else -ABS_CAP
            # Convention: positive funding flows from LONGS to SHORTS.
            # A SHORT receives +qty*mark*rate; a LONG pays the same (negative on LONG's books).
            base = perp_leg.qty * fp.mark_price * settled_rate
            perp_amount = base if perp_leg.side == "short" else -base
            await ctx.storage.credit_funding(perp_leg.id, perp_amount)
            log.info("funding.accrued.perp", symbol=symbol, trade_id=perp_leg.id,
                     amount=perp_amount, rate=settled_rate, side=perp_leg.side)
            # Short-spot borrow cost (only the -1 direction has a SELL spot leg)
            if spot_leg and spot_leg.side == "short":
                borrow_cost = -(spot_leg.qty * fp.mark_price *
                                 (self.p.spot_borrow_bps_per_8h / 10_000))
                await ctx.storage.credit_funding(spot_leg.id, borrow_cost)
                log.info("funding.accrued.borrow", symbol=symbol,
                         trade_id=spot_leg.id, amount=borrow_cost)
            ap.last_credited_funding_ms = fp.next_funding_ms
            try:
                await ctx.storage.update_pair_funding_watermark(
                    ap.pair_id, fp.next_funding_ms,
                )
            except Exception:
                log.exception("funding_harvest.watermark_persist_failed",
                              pair_id=ap.pair_id)

    def _per_pair_notional(self, ctx: StrategyContext) -> float:
        """Scale `notional_per_pair_usd` to fit this strategy's equity slice
        across `max_concurrent_pairs`. F11 + R14: the perp leg only requires
        `notional / leverage` in collateral, so capital required per pair is
        `notional + notional/leverage`, not `2 * notional`.
        """
        configured = self.p.notional_per_pair_usd
        share = ctx.equity_available_usd(self.name)
        # Capital required per pair = spot leg notional + perp margin.
        # If leverage=2: capital_per_unit_notional = 1 + 0.5 = 1.5
        cap_per_unit = 1.0 + (1.0 / max(1, self.p.perp_leverage))
        max_per_pair_from_share = (share / cap_per_unit) / max(1, self.p.max_concurrent_pairs)
        return min(configured, max_per_pair_from_share)

    async def _maybe_open(self, ctx: StrategyContext, symbol: str) -> None:
        cur = self.funding.current_bps(symbol)
        avg = self.funding.avg_bps(symbol, n=21)
        if cur is None or avg is None:
            return
        # Direction:
        #   +1 (positive funding): long_spot + short_perp → perp collects funding
        #   -1 (negative funding): short_spot + long_perp → perp collects (paid)
        # The negative direction needs spot margin enabled; in paper mode we
        # simulate it as "borrow spot, sell, buy-back-to-close" with a borrow fee.
        direction = 0
        if cur >= self.p.entry_threshold_bps and avg >= self.p.entry_avg_threshold_bps:
            direction = 1
        elif cur <= -self.p.entry_threshold_bps and avg <= -self.p.entry_avg_threshold_bps \
                and self.p.allow_negative_direction:
            direction = -1
        if direction == 0:
            return
        ok, why = self.basis.safe_to_open(symbol, self.p.spot_tf_for_basis)
        if not ok:
            log.info("funding_harvest.basis_blocked", symbol=symbol, reason=why,
                     funding_bps=cur)
            return
        fp = self.funding.current(symbol)
        if not fp or fp.mark_price <= 0:
            return
        snap = ctx.indicators.latest(symbol, self.p.spot_tf_for_basis)
        if not snap or snap.close <= 0:
            return
        spot_px = snap.close
        notional = self._per_pair_notional(ctx)
        if notional <= 0:
            log.info("funding_harvest.no_equity_share", symbol=symbol)
            return
        spot_qty = notional / spot_px
        perp_qty = notional / fp.mark_price
        # Build direction-aware pair:
        #   +1: BUY spot + SELL perp  (positive funding → short perp collects)
        #   -1: SELL spot + BUY  perp  (negative funding → long perp collects)
        if direction == 1:
            spot_side, perp_side = "BUY", "SELL"
        else:
            spot_side, perp_side = "SELL", "BUY"
        pair_id = uuid.uuid4().hex[:16]
        pair = PairProposal(
            id=pair_id, strategy=self.name, direction=direction,
            legs=[
                PairLeg(symbol=symbol, market="spot", side=spot_side,
                        qty=spot_qty, expected_price=spot_px, leverage=1),
                PairLeg(symbol=symbol, market="perps", side=perp_side,
                        qty=perp_qty, expected_price=fp.mark_price,
                        leverage=self.p.perp_leverage),
            ],
            notional_usd=notional * 2,
            rationale=f"funding={cur:+.1f}bps (avg21 {avg:+.1f}bps) dir={direction:+d}",
            expected_yield_bps_per_8h=cur if direction == 1 else -cur,
            expires_at_ms=now_ms() + ctx.settings.approval_timeout_sec * 1000,
        )
        log.info("funding_harvest.proposing_open", symbol=symbol,
                 funding_bps=cur, direction=direction, notional=notional)
        await ctx.propose_pair(pair)

    async def _maybe_close(self, ctx: StrategyContext, symbol: str,
                            ap: ActivePair) -> None:
        cur = self.funding.current_bps(symbol)
        if cur is None:
            return
        reason: Optional[str] = None
        # Direction-aware funding-flip exit:
        #   +1 direction wants funding HIGH; close when funding ≤ exit_bps
        #   -1 direction wants funding LOW (very negative); close when funding ≥ -exit_bps
        if ap.direction == 1 and cur <= self.p.exit_threshold_bps:
            reason = "funding_flip"
        elif ap.direction == -1 and cur >= -self.p.exit_threshold_bps:
            reason = "funding_flip"
        else:
            alert = self.basis.needs_exit_alert(symbol, self.p.spot_tf_for_basis)
            if alert is not None:
                reason = "basis_breakout"
        # Perp price-move safety stop (F4) — direction-aware:
        #   +1: short_perp is hurt by price UP
        #   -1: long_perp  is hurt by price DOWN
        if reason is None and ap.perp_entry_mark > 0:
            fp = self.funding.current(symbol)
            if fp and fp.mark_price > 0:
                move_pct = ((fp.mark_price - ap.perp_entry_mark) / ap.perp_entry_mark) * 100.0
                adverse_pct = move_pct if ap.direction == 1 else -move_pct
                if adverse_pct >= self.p.perp_adverse_move_pct:
                    reason = "perp_adverse_move"
                    log.warning("funding_harvest.perp_safety_stop",
                                symbol=symbol, direction=ap.direction,
                                entry=ap.perp_entry_mark, now=fp.mark_price,
                                adverse_pct=adverse_pct)
        if reason is None:
            return
        log.info("funding_harvest.closing", symbol=symbol, reason=reason,
                 funding_bps=cur)
        await self._close_pair(ctx, ap, reason)
        try:
            await ctx.storage.close_pair_meta(ap.pair_id)
        except Exception:
            log.exception("funding_harvest.close_pair_meta_failed",
                          pair_id=ap.pair_id)
        self.active.pop(symbol, None)

    async def _close_pair(self, ctx: StrategyContext, ap: ActivePair, reason: str) -> None:
        """Close both legs through PairExecutor so the perp leg uses perp mark,
        not spot price (F2 fix). Falls back to per-leg close_trade if the
        orchestrator's pair_executor isn't reachable."""
        # Prefer pair_executor.close_pair so the perp leg gets the real mark.
        perp_mark = None
        spot_px = None
        fp = self.funding.current(ap.symbol)
        if fp:
            perp_mark = fp.mark_price
            spot_px = fp.index_price or ctx.last_price.get(ap.symbol)
        # Build the (symbol, market) → price map for PairExecutor.
        price_map: dict[tuple[str, str], float] = {}
        if spot_px:
            price_map[(ap.symbol, "spot")] = spot_px
        if perp_mark:
            price_map[(ap.symbol, "perps")] = perp_mark

        pe = getattr(ctx, "pair_executor", None)
        if pe is not None and price_map:
            # Use exit_reason via the PairExecutor (handles fees+slippage uniformly).
            await pe.close_pair(ap.legs, reason=reason, price_map=price_map)
            return
        # Fallback: per-leg close via orchestrator (uses last_price by symbol)
        for leg in ap.legs:
            await ctx.close_trade(leg.id, reason=reason)

    # Called by orchestrator after a successful propose_pair → fill.
    async def register_active_pair(self, pair: PairProposal, legs: list[Trade]) -> None:
        perp_entry_mark = 0.0
        for leg in pair.legs:
            if leg.market == "perps":
                perp_entry_mark = leg.expected_price
                break
        direction = pair.direction
        symbol = pair.legs[0].symbol
        opened_ms = now_ms()
        self.active[symbol] = ActivePair(
            pair_id=pair.id, symbol=symbol,
            legs=list(legs), opened_ms=opened_ms, direction=direction,
            perp_entry_mark=perp_entry_mark,
        )
        # R6: persist pair metadata so the direction + perp_entry_mark + funding
        # watermark survive a restart.
        if self.ctx is not None:
            try:
                await self.ctx.storage.save_pair_meta(
                    pair_id=pair.id, strategy=self.name, symbol=symbol,
                    direction=direction, perp_entry_mark=perp_entry_mark,
                    opened_ms=opened_ms,
                )
            except Exception:
                log.exception("funding_harvest.save_pair_meta_failed", pair_id=pair.id)
