"""Deterministic risk gate. The LLM cannot bypass this; even Telegram-approved
trades must pass `check()`. Two responsibilities:

  1. Reject proposals that violate guardrails (daily loss, leverage, count,
     correlation, cost-of-edge, liquidation distance).
  2. Size the position: convert (entry, stop, equity, risk_pct) → qty.

State (PnL today, consecutive losses, open positions) is supplied by the
caller — this module is intentionally stateless so it stays unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from src.config.settings import Settings, get_settings
from src.models.types import Position, Signal

log = structlog.get_logger(__name__)


@dataclass
class AccountState:
    equity_usd: float
    pnl_today_usd: float
    consecutive_losses: int
    open_positions: list[Position]
    last_trade_ms_by_symbol: dict[str, int]
    halted_until_ms: int = 0
    # Optional: per-symbol BTC-beta from CorrelationMatrix. None → fall back
    # to a 0.85 default in `_btc_beta_equiv`.
    btc_betas: Optional[dict[str, float]] = None


@dataclass
class RiskDecision:
    ok: bool
    qty: float = 0.0
    notional_usd: float = 0.0
    leverage: int = 1
    reason: str = ""


def _exposure_per_coin(positions: list[Position], symbol: str) -> float:
    return sum(p.qty * p.entry for p in positions if p.symbol == symbol)


def _btc_beta_equiv(positions: list[Position],
                    betas: Optional[dict[str, float]] = None) -> float:
    """BTC-beta-weighted total exposure. Uses live `betas` if provided
    (from CorrelationMatrix); falls back to a 0.85 default per symbol if
    no estimate is available. J1 wires the live matrix."""
    total = 0.0
    for p in positions:
        beta = (betas or {}).get(p.symbol, 0.85)
        total += p.qty * p.entry * beta
    return total


class RiskGate:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.s = settings or get_settings()

    def position_size(self, signal: Signal, equity_usd: float,
                      max_notional_override: Optional[float] = None
                      ) -> tuple[float, float]:
        """Return (qty, notional_usd). ATR-based fixed-fractional risk.
        `max_notional_override` (if provided) tightens the per-trade cap
        below `settings.max_notional_usd`. R7: used by NotionalRamp so we
        don't mutate the shared settings object."""
        cap = self.s.max_notional_usd if max_notional_override is None \
            else min(max_notional_override, self.s.max_notional_usd)
        risk_usd = equity_usd * (self.s.risk_per_trade_pct / 100.0)
        risk_per_unit = abs(signal.entry - signal.stop)
        if risk_per_unit <= 0:
            return 0.0, 0.0
        qty = risk_usd / risk_per_unit
        notional = qty * signal.entry
        if notional > cap:
            scale = cap / notional
            qty *= scale
            notional = qty * signal.entry
        return qty, notional

    def check(self, signal: Signal, acct: AccountState, now_ms: int,
              market: str, leverage: int,
              max_notional_override: Optional[float] = None) -> RiskDecision:
        s = self.s

        if now_ms < acct.halted_until_ms:
            return RiskDecision(ok=False, reason=f"halted until {acct.halted_until_ms}")

        if acct.pnl_today_usd <= -(acct.equity_usd * s.max_daily_loss_pct / 100.0):
            return RiskDecision(ok=False, reason="daily loss cap")

        if len(acct.open_positions) >= s.max_concurrent_positions:
            return RiskDecision(ok=False, reason="max concurrent positions")

        if market == "perps" and leverage > s.max_leverage:
            return RiskDecision(ok=False, reason=f"leverage {leverage}x > max {s.max_leverage}x")

        if acct.consecutive_losses >= s.consecutive_loss_halt_long:
            return RiskDecision(ok=False, reason="tilt: long halt")
        if acct.consecutive_losses >= s.consecutive_loss_halt_short:
            return RiskDecision(ok=False, reason="tilt: short halt")

        last = acct.last_trade_ms_by_symbol.get(signal.symbol, 0)
        if last and now_ms - last < s.min_gap_between_trades_sec * 1000:
            return RiskDecision(ok=False, reason="min gap between trades")

        qty, notional = self.position_size(signal, acct.equity_usd,
                                           max_notional_override)
        if qty <= 0 or notional <= 0:
            return RiskDecision(ok=False, reason="zero qty after sizing")

        # Per-coin exposure cap
        per_coin_cap = acct.equity_usd * (s.max_exposure_per_coin_pct / 100.0)
        existing_exposure = _exposure_per_coin(acct.open_positions, signal.symbol)
        if existing_exposure + notional > per_coin_cap:
            return RiskDecision(ok=False, reason="per-coin exposure cap")

        # Correlation: BTC-beta-equivalent total exposure cap, as % of equity.
        sym_beta = (acct.btc_betas or {}).get(signal.symbol, 0.85)
        beta_equiv = _btc_beta_equiv(acct.open_positions, acct.btc_betas) \
            + notional * sym_beta
        beta_cap = acct.equity_usd * (s.max_correlated_exposure_pct / 100.0)
        if beta_equiv > beta_cap:
            return RiskDecision(
                ok=False,
                reason=f"correlation: BTC-beta exposure ${beta_equiv:.0f} > cap ${beta_cap:.0f}",
            )

        # Cost-of-edge filter
        if signal.edge_bps <= 0:
            return RiskDecision(ok=False, reason=f"edge {signal.edge_bps:.1f}bps after costs")

        # Liquidation distance for perps
        if market == "perps" and leverage > 1:
            liq_pct = 1.0 / leverage * 0.95  # approximate, ignores MMR tier
            risk_pct = abs(signal.entry - signal.stop) / signal.entry
            if risk_pct >= liq_pct:
                return RiskDecision(
                    ok=False,
                    reason=f"stop ({risk_pct:.2%}) too close to liq ({liq_pct:.2%}) at {leverage}x",
                )

        return RiskDecision(ok=True, qty=qty, notional_usd=notional, leverage=leverage)
