"""Delta-neutral basis book with an automatic funding-regime gate.

Harvests perp funding with ~zero price risk (long spot + short perp of the same
asset), but ONLY when funding clears the cost hurdle; otherwise sits in USD. The
gate switches regime automatically off a trailing funding signal — see
`research/portfolio/BASIS_SPEC.md` for the full design + 3y validation
(GATED +7.4%/yr vs always-on +5.6% vs always-USD +4.5%).

Thresholds are ANCHORED TO THE ECONOMICS, not tuned: the basis nets above USD
only when gross funding clears USD_yield (~4.5%) + borrow/exec (~2.5%) ≈ 7%/yr.

**Phasing (safety):** `execute_legs=False` (default) runs the strategy in
MONITOR mode — it constantly checks funding, flips the ON/OFF regime with
hysteresis, persists the state, and ALERTS on every transition, but does NOT
place orders. Leg execution (Phase 2) needs the spot+perp executor + testnet
validation and is gated behind `execute_legs=True`. Monitor mode is safe to run
in prod today: it tells you exactly when the regime turns ON so you never miss a
funding spike, with zero execution risk.

The gate math (`compute_signal`, `update_regime`, `included_names`) is pure and
unit-tested; the class is the live wrapper.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import structlog

from src.models.types import now_ms
from src.strategies.base import Strategy, StrategyContext

log = structlog.get_logger(__name__)

DAY_MS = 86_400_000
# Persistence-screened basket: funding positive EVERY year 2023-2026. Majors are
# the low-tail anchors; the alts add yield. Names that flipped funding-negative
# in any year (SOL/XRP/FIL/DOT/ATOM/INJ/BNB) are deliberately excluded.
DEFAULT_BASKET = ("BTCUSDT", "ETHUSDT", "LINKUSDT", "UNIUSDT",
                  "LTCUSDT", "DOGEUSDT", "AAVEUSDT")


@dataclass(frozen=True)
class BasisParams:
    basket: tuple[str, ...] = DEFAULT_BASKET
    lookback_days: int = 21               # trailing window for the funding signal
    usd_yield_pct: float = 4.5            # stablecoin/T-bill baseline
    borrow_exec_pct: float = 2.5          # basis carrying cost
    on_margin_pct: float = 2.0            # hysteresis band above the hurdle
    check_interval_s: int = 6 * 3600      # re-evaluate every 6h (funding is slow)
    execute_legs: bool = False            # Phase 2 gate — False = monitor only

    @property
    def hurdle_pct(self) -> float:
        """OFF line: basis-net == USD when gross funding == this."""
        return self.usd_yield_pct + self.borrow_exec_pct

    @property
    def on_pct(self) -> float:
        """ON line (hurdle + hysteresis margin)."""
        return self.hurdle_pct + self.on_margin_pct


# ─────────────────────────────── pure gate logic ───────────────────────────

def _annualized_pct(rates: list[float]) -> float:
    """Mean 8h funding rate → annualized % (3 settlements/day × 365)."""
    if not rates:
        return 0.0
    return sum(rates) / len(rates) * 3 * 365 * 100.0


def compute_signal(funding: dict[str, list[tuple[int, float]]], now: int,
                   lookback_days: int) -> tuple[float, dict[str, float]]:
    """(basket_signal_pct, {symbol: per_name_ann_pct}). PIT: only funding events
    in the trailing `lookback_days` strictly before `now`."""
    lo = now - lookback_days * DAY_MS
    per_name: dict[str, float] = {}
    for sym, evs in funding.items():
        rates = [r for (t, r) in evs if lo <= t < now]
        if rates:
            per_name[sym] = _annualized_pct(rates)
    basket = sum(per_name.values()) / len(per_name) if per_name else 0.0
    return basket, per_name


def update_regime(current_on: bool, basket_sig: float,
                  on_pct: float, off_pct: float) -> bool:
    """Hysteresis state machine: ON when signal ≥ on_pct, OFF when < off_pct,
    else hold current state (prevents whipsaw around the hurdle)."""
    if not current_on and basket_sig >= on_pct:
        return True
    if current_on and basket_sig < off_pct:
        return False
    return current_on


def included_names(per_name: dict[str, float], hurdle_pct: float) -> list[str]:
    """Names to actually hold when ON — only those whose own trailing funding
    clears the per-name hurdle (drop any gone lean/negative)."""
    return sorted(s for s, v in per_name.items() if v >= hurdle_pct)


# ───────────────────────────────── strategy ────────────────────────────────

class BasisStrategy(Strategy):
    """Funding-gated delta-neutral basis book. Phase 1 = monitor + auto-switch."""

    name = "basis"

    def __init__(self, params: Optional[BasisParams] = None) -> None:
        self.p = params or BasisParams()
        self.ctx: Optional[StrategyContext] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_ev = asyncio.Event()
        self.regime_on: bool = False

    async def start(self, ctx: StrategyContext) -> None:
        self.ctx = ctx
        try:
            row = await ctx.storage.load_strategy_state(self.name)
            if row and "regime_on" in row:
                self.regime_on = bool(row["regime_on"])
        except Exception:
            log.warning("basis.state_load_failed_using_off")
        self._stop_ev.clear()
        self._task = asyncio.create_task(self._loop())
        log.info("basis.started", basket=len(self.p.basket),
                 on_pct=self.p.on_pct, off_pct=self.p.hurdle_pct,
                 execute_legs=self.p.execute_legs, regime_on=self.regime_on)

    async def stop(self) -> None:
        self._stop_ev.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        # Evaluate once at startup, then every check_interval.
        while not self._stop_ev.is_set():
            try:
                await self._evaluate()
            except Exception:
                log.exception("basis.evaluate_failed")
            try:
                await asyncio.wait_for(self._stop_ev.wait(),
                                       timeout=self.p.check_interval_s)
            except asyncio.TimeoutError:
                pass

    async def _fetch_funding(self, sym: str, start_ms: int, end_ms: int
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
                    log.warning("basis.funding_fetch_failed", symbol=sym, err=str(e))
                    return out
            if not page:
                break
            for row in page:
                out.append((int(row["fundingTime"]), float(row["fundingRate"])))
            last = int(page[-1]["fundingTime"])
            if len(page) < 1000 or last <= cursor:
                break
            cursor = last + 1
        return out

    async def _evaluate(self) -> None:
        t = now_ms()
        start = t - self.p.lookback_days * DAY_MS
        funding: dict[str, list[tuple[int, float]]] = {}
        for sym in self.p.basket:
            funding[sym] = await self._fetch_funding(sym, start, t)

        basket_sig, per_name = compute_signal(funding, t, self.p.lookback_days)
        prev = self.regime_on
        self.regime_on = update_regime(prev, basket_sig,
                                       self.p.on_pct, self.p.hurdle_pct)
        incl = included_names(per_name, self.p.hurdle_pct) if self.regime_on else []

        log.info("basis.evaluated", basket_signal_pct=round(basket_sig, 2),
                 regime="ON" if self.regime_on else "OFF",
                 on_pct=self.p.on_pct, off_pct=self.p.hurdle_pct,
                 included=incl, execute_legs=self.p.execute_legs)

        await self._persist()

        if self.regime_on != prev:
            await self._on_transition(basket_sig, incl)
        # Phase 2: when execute_legs and regime_on → open/rebalance basis legs;
        #          when execute_legs and just turned OFF → unwind to USD.
        # Monitor mode (Phase 1) stops here: state tracked + alerted, no orders.

    async def _on_transition(self, sig: float, incl: list[str]) -> None:
        direction = "ON — funding pays" if self.regime_on else "OFF — hold USD"
        msg = (f"*Basis regime → {direction}*\n"
               f"  trailing {self.p.lookback_days}d basket funding: {sig:.1f}%/yr\n"
               f"  thresholds: ON≥{self.p.on_pct:.0f}% / OFF<{self.p.hurdle_pct:.0f}%")
        if self.regime_on:
            msg += f"\n  would deploy basis on: {', '.join(incl) or '(none clear hurdle)'}"
        if not self.p.execute_legs:
            msg += "\n  _(monitor mode: no orders placed)_"
        log.info("basis.regime_transition", regime_on=self.regime_on,
                 signal_pct=round(sig, 2), included=incl)
        if getattr(self.ctx, "telegram", None):
            try:
                await self.ctx.telegram.send_info(msg)  # type: ignore[union-attr]
            except Exception:
                log.warning("basis.telegram_alert_failed")
        await self.ctx.storage.audit("basis_regime_transition", {
            "regime_on": self.regime_on, "signal_pct": sig, "included": incl,
        })

    async def _persist(self) -> None:
        try:
            await self.ctx.storage.save_strategy_state(
                self.name, {"regime_on": self.regime_on})
        except Exception:
            log.warning("basis.state_persist_failed")
