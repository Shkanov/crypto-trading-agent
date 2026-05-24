"""Adaptive notional sizing.

Goal: start tiny ($25 max notional), scale up gradually only when the
strategy has demonstrated real edge. Halve on a 3% drawdown day so
losses can't compound.

State persisted to storage so a restart doesn't reset the ramp. The
RiskGate reads `effective_max_notional_usd` instead of the static
`settings.max_notional_usd` cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import structlog

from src.services.storage import Storage

log = structlog.get_logger(__name__)


@dataclass
class RampState:
    current_max_notional_usd: float
    week_start_pnl_usd: float = 0.0
    week_start_ts_ms: int = 0
    last_review_ts_ms: int = 0
    consecutive_profitable_weeks: int = 0


class NotionalRamp:
    """Ramps `max_notional_usd` based on weekly performance.

    Policy:
      - Initialize at `starting_notional_usd` (override of settings.max_notional_usd).
      - At each weekly review (7d since last_review), if pnl since last_review > 0:
          new_max = current_max * (1 + step_pct)
          consecutive_profitable_weeks += 1
      - On a daily realized loss > drawdown_halve_pct of equity:
          new_max = current_max / 2
          consecutive_profitable_weeks = 0
      - Hard floor at `min_notional_usd`; hard ceiling at `max_notional_usd_ceiling`.
    """

    def __init__(self, storage: Storage,
                 starting_notional_usd: float = 25.0,
                 step_pct: float = 10.0,
                 drawdown_halve_pct: float = 3.0,
                 min_notional_usd: float = 10.0,
                 max_notional_usd_ceiling: float = 2000.0) -> None:
        self.storage = storage
        self.step_pct = step_pct
        self.drawdown_halve_pct = drawdown_halve_pct
        self.min_notional = min_notional_usd
        self.ceiling = max_notional_usd_ceiling
        self.state = RampState(current_max_notional_usd=starting_notional_usd)

    async def load(self) -> None:
        """Load persisted ramp state from storage. Idempotent — defaults to
        the constructor-time `starting_notional_usd` if nothing persisted."""
        loaded = await self.storage.load_ramp_state()
        if loaded is not None:
            self.state.current_max_notional_usd = loaded[0]
            self.state.last_review_ts_ms = loaded[1]
            self.state.consecutive_profitable_weeks = loaded[2]
            log.info("notional_ramp.loaded",
                     max_notional=self.state.current_max_notional_usd,
                     last_review_ms=self.state.last_review_ts_ms,
                     streak_weeks=self.state.consecutive_profitable_weeks)

    async def _persist(self) -> None:
        await self.storage.save_ramp_state(
            self.state.current_max_notional_usd,
            self.state.last_review_ts_ms,
            self.state.consecutive_profitable_weeks,
        )

    @property
    def effective_max_notional_usd(self) -> float:
        return self.state.current_max_notional_usd

    async def weekly_review(self, pnl_since_last_review_usd: float,
                            now_ms: int) -> Optional[str]:
        if self.state.last_review_ts_ms == 0:
            self.state.last_review_ts_ms = now_ms
            return None
        if now_ms - self.state.last_review_ts_ms < 7 * 86_400 * 1000:
            return None
        msg: Optional[str] = None
        if pnl_since_last_review_usd > 0:
            old = self.state.current_max_notional_usd
            new = min(self.ceiling, old * (1 + self.step_pct / 100.0))
            self.state.current_max_notional_usd = new
            self.state.consecutive_profitable_weeks += 1
            msg = (f"Ramp up: profitable week ({pnl_since_last_review_usd:+.2f}) — "
                   f"max_notional {old:.0f} → {new:.0f} "
                   f"(streak: {self.state.consecutive_profitable_weeks})")
        else:
            self.state.consecutive_profitable_weeks = 0
            msg = (f"Ramp hold: unprofitable week ({pnl_since_last_review_usd:+.2f}) "
                   f"— max_notional unchanged at {self.state.current_max_notional_usd:.0f}")
        self.state.last_review_ts_ms = now_ms
        await self.storage.audit("notional_ramp.weekly", {
            "pnl": pnl_since_last_review_usd,
            "new_max_notional": self.state.current_max_notional_usd,
            "streak_weeks": self.state.consecutive_profitable_weeks,
        })
        await self._persist()
        log.info("notional_ramp.weekly", **{"pnl": pnl_since_last_review_usd,
                                            "new_max": self.state.current_max_notional_usd})
        return msg

    async def drawdown_check(self, pnl_today_usd: float, equity_usd: float,
                              now_ms: int) -> Optional[str]:
        """Call from housekeeping. Halves if today's loss exceeds threshold."""
        if equity_usd <= 0:
            return None
        loss_pct = -(pnl_today_usd / equity_usd) * 100.0
        if loss_pct < self.drawdown_halve_pct:
            return None
        old = self.state.current_max_notional_usd
        new = max(self.min_notional, old / 2.0)
        if new >= old:
            return None
        self.state.current_max_notional_usd = new
        self.state.consecutive_profitable_weeks = 0
        await self.storage.audit("notional_ramp.drawdown_halve", {
            "pnl_today": pnl_today_usd, "loss_pct": loss_pct,
            "old_max": old, "new_max": new,
        })
        await self._persist()
        log.warning("notional_ramp.halved", pnl_today=pnl_today_usd,
                    old_max=old, new_max=new)
        return f"Drawdown halve: {loss_pct:.2f}% loss today → max_notional {old:.0f} → {new:.0f}"
