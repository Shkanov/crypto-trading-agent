"""BTC HODL benchmark — the honest yardstick.

Snapshot equity + BTC price on first call. Every subsequent call computes:
  hodl_equity = equity_at_start * (btc_now / btc_at_start)
  outperformance = current_equity - hodl_equity

The point: a strategy that "made $20 last week" while BTC went up 8% is
not a winning strategy — it underperformed the trivial benchmark. We
surface this on /status and daily digests so the operator can't pretend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


@dataclass
class HodlSnapshot:
    equity_at_start: float
    btc_at_start: float
    started_ms: int


class HodlBenchmark:
    def __init__(self) -> None:
        self.snap: Optional[HodlSnapshot] = None

    def initialize(self, equity_usd: float, btc_price: float, ts_ms: int) -> None:
        if self.snap is not None:
            return
        if equity_usd <= 0 or btc_price <= 0:
            return
        self.snap = HodlSnapshot(
            equity_at_start=equity_usd, btc_at_start=btc_price, started_ms=ts_ms,
        )
        log.info("hodl.initialized", equity=equity_usd, btc=btc_price)

    def hodl_equity(self, btc_now: float) -> Optional[float]:
        if self.snap is None or btc_now <= 0:
            return None
        return self.snap.equity_at_start * (btc_now / self.snap.btc_at_start)

    def outperformance_usd(self, current_equity: float, btc_now: float
                            ) -> Optional[float]:
        hodl = self.hodl_equity(btc_now)
        return None if hodl is None else current_equity - hodl

    def outperformance_pct(self, current_equity: float, btc_now: float
                            ) -> Optional[float]:
        hodl = self.hodl_equity(btc_now)
        if hodl is None or hodl <= 0:
            return None
        return ((current_equity / hodl) - 1.0) * 100.0
