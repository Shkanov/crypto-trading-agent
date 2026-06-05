"""Live Δfunding cross-sectional carry strategy.

Validated CPCV edge: PASS on top-30 (w21, OOS +0.978, PBO 0.429) and
top-50 (w42, OOS +0.451, PBO 0.000). Alpha comes from price drift that
follows funding repricing — NOT from collecting yield (both legs pay
funding; the first-difference of a near-unit-root funding series captures
repricing events before the market fully adjusts).

Mechanics:
  Every `rebalance_hours` (default 168h = 1 week):
    1. Build a top-N USDT perp universe by 24h volume (ex-majors, stables).
    2. For each symbol compute Δfunding = mean(funding, recent window) −
       mean(funding, prior window).  Uses `funding_carry.funding_window_change`.
    3. LONG top-N by Δfunding (funding accelerating), SHORT bottom-N.
    4. Close any positions from the PREVIOUS basket not in the new one.
    5. Open new single-leg perp positions for each basket member.

Sizing: per-leg notional = equity_available × book_pct_per_side / top_n.
The stop price is back-calculated so the orchestrator's risk-pct sizing
produces exactly that notional (stop_pct = risk_per_trade_pct/100 ×
equity / notional_per_leg).  The stop is wide (~30% at default settings)
and serves as a hard safety backstop, not an expected exit; normal exits
happen on the next rebalance.

Enable: set `dfunding_carry_enabled = true` in .env / settings.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import structlog

from src.models.types import (
    FeatureVector,
    Signal,
    Trade,
    now_ms,
    stable_proposal_id,
)
from src.strategies.base import Strategy, StrategyContext
from src.strategies.funding_carry import (
    CarryParams,
    build_rebalance,
    funding_window_change,
    rank_for_carry,
)

log = structlog.get_logger(__name__)

EIGHT_H_MS = 8 * 3_600_000
STABLE_TOKENS = ("FDUSD", "USDC", "EUR", "USD1", "TUSD", "BUSD", "DAI", "PYUSD")
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BEARUSDT", "BULLUSDT")
MAJORS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"})


@dataclass(frozen=True)
class DFundingCarryParams:
    window_cycles: int = 21          # funding cycles per window; 21×8h = 168h
    top_n: int = 3                   # legs per side (longs + shorts)
    book_pct_per_side: float = 0.10  # fraction of available equity per leg-bundle
    universe_size: int = 30          # top-N perps by 24h volume
    rebalance_hours: int = 168       # 7 days between rebalances
    check_interval_s: int = 300      # how often the strategy wakes to check

    @property
    def window_hours(self) -> int:
        return self.window_cycles * 8

    def to_carry_params(self) -> CarryParams:
        return CarryParams(top_n=self.top_n,
                           book_pct_per_side=self.book_pct_per_side)


class DFundingCarryStrategy(Strategy):
    """Weekly-rebalance Δfunding cross-sectional perp strategy."""

    name = "dfunding"

    def __init__(self, params: Optional[DFundingCarryParams] = None) -> None:
        self.p = params or DFundingCarryParams()
        self.ctx: Optional[StrategyContext] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_ev = asyncio.Event()
        self.last_rebalance_ms: int = 0

    # ------------------------------------------------------------------ lifecycle

    async def start(self, ctx: StrategyContext) -> None:
        self.ctx = ctx
        # Attempt to recover last_rebalance_ms from the audit log so a restart
        # doesn't immediately trigger a redundant rebalance.
        try:
            row = await ctx.storage.load_strategy_state(self.name)
            if row and row.get("last_rebalance_ms"):
                self.last_rebalance_ms = int(row["last_rebalance_ms"])
        except Exception:
            log.warning("dfunding.state_load_failed_using_zero")
        self._stop_ev.clear()
        self._task = asyncio.create_task(self._loop())
        log.info("dfunding.started",
                 window_cycles=self.p.window_cycles,
                 top_n=self.p.top_n,
                 universe=self.p.universe_size,
                 rebalance_h=self.p.rebalance_hours)

    async def stop(self) -> None:
        self._stop_ev.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------ main loop

    async def _loop(self) -> None:
        while not self._stop_ev.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_ev.wait(),
                    timeout=self.p.check_interval_s,
                )
            except asyncio.TimeoutError:
                pass
            if self._stop_ev.is_set():
                break
            elapsed = now_ms() - self.last_rebalance_ms
            if elapsed >= self.p.rebalance_hours * 3_600_000:
                try:
                    await self._rebalance()
                except Exception:
                    log.exception("dfunding.rebalance_failed")

    # ------------------------------------------------------------------ signal

    async def _build_universe(self) -> list[str]:
        b = self.ctx.binance
        assert b.client is not None
        await b.respect_ban()
        async with b.rest_limiter:
            tickers = await b.client.futures_ticker()
        # Only genuinely tradeable USDT crypto perps: intersect with the loaded
        # perp filters (excludes Binance TRADIFI_PERPETUAL tokenized-equity perps
        # like MRVLUSDT/SOXLUSDT/XAUUSDT, which are not in perp_filters and fail
        # quantize), and drop names whose min-notional exceeds our per-leg size
        # (e.g. BCHUSDT minNotional=20 vs a ~$13 leg).
        per_leg = (self.ctx.equity_available_usd(self.name)
                   * self.p.book_pct_per_side / max(1, self.p.top_n))
        rows = []
        for r in tickers:
            sym = r["symbol"]
            if not sym.endswith("USDT"):
                continue
            if sym in MAJORS:
                continue
            if any(tok in sym[:-4] for tok in STABLE_TOKENS):
                continue
            if any(sym.endswith(suf) for suf in LEVERAGED_SUFFIXES):
                continue
            filt = b.perp_filters.get(sym)
            if filt is None:
                continue
            if per_leg > 0 and float(filt.min_notional) > per_leg:
                continue
            rows.append(r)
        rows.sort(key=lambda r: float(r["quoteVolume"]), reverse=True)
        return [r["symbol"] for r in rows[: self.p.universe_size]]

    async def _fetch_funding(self, sym: str,
                              start_ms: int, end_ms: int
                              ) -> list[tuple[int, float]]:
        b = self.ctx.binance
        assert b.client is not None
        out: list[tuple[int, float]] = []
        cursor = start_ms
        while cursor < end_ms:
            await b.respect_ban()
            async with b.rest_limiter:
                try:
                    page = await b.client.futures_funding_rate(
                        symbol=sym, startTime=cursor, endTime=end_ms, limit=1000)
                except Exception as e:  # noqa: BLE001
                    log.warning("dfunding.funding_fetch_failed",
                                symbol=sym, err=str(e))
                    return out
            if not page:
                break
            for row in page:
                out.append((int(row["fundingTime"]), float(row["fundingRate"])))
            last_t = int(page[-1]["fundingTime"])
            if len(page) < 1000 or last_t <= cursor:
                break
            cursor = last_t + 1
        out.sort(key=lambda x: x[0])
        return out

    async def _compute_signal(self) -> dict[str, float]:
        """Build universe + compute per-symbol Δfunding. PIT-safe: uses only
        funding events strictly before now_ms."""
        t_now = now_ms()
        start_ms = t_now - 2 * self.p.window_hours * 3_600_000

        universe = await self._build_universe()
        log.info("dfunding.universe_built", n=len(universe),
                 syms=universe[:5])

        tasks = {sym: asyncio.create_task(
            self._fetch_funding(sym, start_ms, t_now)) for sym in universe}
        signal: dict[str, float] = {}
        for sym, task in tasks.items():
            funding = await task
            df = funding_window_change(funding, t_now, self.p.window_hours)
            if df is not None:
                signal[sym] = df
        log.info("dfunding.signal_computed",
                 n_symbols=len(signal), window_h=self.p.window_hours)
        return signal

    # ------------------------------------------------------------------ rebalance

    async def _get_mark_price(self, sym: str) -> Optional[float]:
        """Fetch current mark price for a perp symbol."""
        # Try the orchestrator's price cache first (populated by market loop).
        cached = self.ctx.last_price.get(sym)
        if cached and cached > 0:
            return cached
        # Fall back to a REST call for symbols outside the main symbol_list.
        b = self.ctx.binance
        assert b.client is not None
        try:
            async with b.rest_limiter:
                tick = await b.client.futures_symbol_ticker(symbol=sym)
            return float(tick["price"])
        except Exception as e:  # noqa: BLE001
            log.warning("dfunding.mark_price_fetch_failed", symbol=sym, err=str(e))
            return None

    async def _open_perp(self, sym: str, side: str, notional: float,
                          mark_price: float) -> None:
        """Submit a single-leg perp proposal sized to `notional` USD."""
        ctx = self.ctx
        s = ctx.settings
        # Back-calculate stop_pct so risk-sizing gives exactly `notional`.
        # qty = risk_usd / (stop_pct × mark), notional = qty × mark
        # → stop_pct = risk_usd / notional
        equity = ctx.equity_available_usd(self.name)
        risk_usd = equity * s.risk_per_trade_pct / 100.0
        stop_pct = (risk_usd / notional) if notional > 0 else 0.30
        stop_pct = max(0.05, min(stop_pct, 0.50))  # clamp 5-50%

        if side == "long":
            stop = mark_price * (1.0 - stop_pct)
            take_profit = mark_price * (1.0 + 3.0 * stop_pct)
        else:
            stop = mark_price * (1.0 + stop_pct)
            take_profit = mark_price * (1.0 - 3.0 * stop_pct)

        signal = Signal(
            id=stable_proposal_id(sym, side,
                                   f"dfunding_{self.last_rebalance_ms}_{sym}"),
            symbol=sym,
            side=side,
            confidence=0.7,
            score=0.7,
            entry=mark_price,
            stop=stop,
            take_profit=take_profit,
            edge_bps=50.0,
            features=FeatureVector(trend=0.3, volume=0.3),
            rationale=(f"dfunding rebalance w{self.p.window_cycles} "
                       f"top{self.p.top_n}"),
        )
        await ctx.propose(signal, market="perps", leverage=1,
                           strategy_name=self.name)
        log.info("dfunding.position_proposed",
                 symbol=sym, side=side, notional=f"${notional:.0f}",
                 mark=f"${mark_price:.4f}")

    async def _rebalance(self) -> None:
        ctx = self.ctx
        t_now = now_ms()

        signal = await self._compute_signal()
        if len(signal) < 2 * self.p.top_n:
            log.warning("dfunding.insufficient_signal_coverage",
                        have=len(signal), need=2 * self.p.top_n)
            return

        equity = ctx.equity_available_usd(self.name)
        rb = build_rebalance(signal, equity_usd=equity,
                              ts_ms=t_now, p=self.p.to_carry_params())
        if not rb.is_active:
            log.warning("dfunding.empty_rebalance", reason=rb.skipped_reason)
            return

        new_longs = {pos.symbol for pos in rb.longs}
        new_shorts = {pos.symbol for pos in rb.shorts}
        current: list[Trade] = ctx.open_trades(self.name)
        current_longs = {t.symbol for t in current if t.side == "long"}
        current_shorts = {t.symbol for t in current if t.side == "short"}

        # ── Close stale positions ──
        to_close = [
            t for t in current
            if (t.side == "long" and t.symbol not in new_longs)
            or (t.side == "short" and t.symbol not in new_shorts)
        ]
        for t in to_close:
            try:
                await ctx.close_trade(t.id, "dfunding_rebalance")
                log.info("dfunding.position_closed",
                         symbol=t.symbol, side=t.side, id=t.id)
            except Exception:
                log.exception("dfunding.close_failed",
                               symbol=t.symbol, id=t.id)

        if to_close:
            # Brief pause to let closes settle before opening new legs.
            await asyncio.sleep(3.0)

        # ── Open new positions ──
        per_leg = equity * self.p.book_pct_per_side / self.p.top_n
        for pos in rb.longs:
            if pos.symbol in current_longs:
                continue
            mark = await self._get_mark_price(pos.symbol)
            if mark is None:
                continue
            try:
                await self._open_perp(pos.symbol, "long", per_leg, mark)
            except Exception:
                log.exception("dfunding.open_long_failed", symbol=pos.symbol)

        for pos in rb.shorts:
            if pos.symbol in current_shorts:
                continue
            mark = await self._get_mark_price(pos.symbol)
            if mark is None:
                continue
            try:
                await self._open_perp(pos.symbol, "short", per_leg, mark)
            except Exception:
                log.exception("dfunding.open_short_failed", symbol=pos.symbol)

        self.last_rebalance_ms = t_now

        # Persist for crash-recovery.
        try:
            await ctx.storage.save_strategy_state(
                self.name, {"last_rebalance_ms": t_now})
        except Exception:
            log.warning("dfunding.state_persist_failed")

        await ctx.storage.audit("dfunding_rebalanced", {
            "new_longs": sorted(new_longs),
            "new_shorts": sorted(new_shorts),
            "closed": [t.symbol for t in to_close],
            "equity_available": equity,
            "per_leg_notional": per_leg,
        })
        log.info("dfunding.rebalanced",
                 longs=sorted(new_longs), shorts=sorted(new_shorts),
                 closed=len(to_close), equity=f"${equity:.0f}")

        if hasattr(ctx, "telegram") and ctx.telegram:  # type: ignore[union-attr]
            lines = ["*Δfunding rebalanced*",
                     f"  longs: {', '.join(sorted(new_longs))}",
                     f"  shorts: {', '.join(sorted(new_shorts))}"]
            if to_close:
                lines.append(f"  closed: {', '.join(t.symbol for t in to_close)}")
            await ctx.telegram.send_info("\n".join(lines))  # type: ignore[union-attr]
