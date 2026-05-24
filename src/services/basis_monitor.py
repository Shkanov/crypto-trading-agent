"""Basis monitor for delta-neutral pair trades.

When you're long spot + short perp (or the inverse), your "delta neutrality"
only holds while spot and perp track each other tightly. During panics, perp
can dislocate 1-3% from spot for hours — at that point you ARE directional,
silently. This module surfaces basis blowups so the strategy can:
  - PAUSE new pair entries while basis is unhealthy
  - ALERT on existing pairs when basis exceeds an exit threshold

Basis convention used here: `(perp_mark - spot_mid) / spot_mid` in bps.
Positive basis = perp trading rich vs spot (the normal regime when funding > 0).
Negative basis = perp discount (rare; usually accompanies sustained negative funding).

Reads spot mid from the IndicatorEngine's latest close (proxy — fine for
intra-day basis monitoring; for sub-second precision you'd want top-of-book).
Reads perp mark from FundingMonitor (mark price comes free with premium index).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from src.services.funding_monitor import FundingMonitor
from src.tools.indicators import IndicatorEngine

log = structlog.get_logger(__name__)


@dataclass
class BasisSample:
    symbol: str
    spot: float
    perp_mark: float
    basis_bps: float


class BasisMonitor:
    def __init__(self, funding: FundingMonitor, indicators: IndicatorEngine,
                 entry_block_bps: float = 50.0,
                 exit_alert_bps: float = 150.0) -> None:
        self.funding = funding
        self.indicators = indicators
        # Hard band: refuse to open NEW pairs while |basis| > entry_block_bps.
        self.entry_block_bps = entry_block_bps
        # Soft band: alert on existing pairs when |basis| > exit_alert_bps.
        self.exit_alert_bps = exit_alert_bps

    def sample(self, symbol: str, spot_tf: str = "5m") -> Optional[BasisSample]:
        """Sample basis = (perp_mark - spot_ref) / spot_ref.

        F14: prefer FundingMonitor's `index_price` (real-time from
        /premiumIndex) over the 5m kline close, which can be up to ~5 min
        stale. The kline close is kept as a fallback when index_price is
        missing (early in startup or thin venues)."""
        fp = self.funding.current(symbol)
        if not fp or fp.mark_price <= 0:
            return None
        # Preferred: real-time index price from premiumIndex.
        spot = fp.index_price if fp.index_price and fp.index_price > 0 else None
        if spot is None:
            snap = self.indicators.latest(symbol, spot_tf)
            spot = snap.close if snap else None
        if spot is None:
            return None
        basis_bps = ((fp.mark_price - spot) / spot) * 10_000
        return BasisSample(symbol=symbol, spot=spot, perp_mark=fp.mark_price,
                           basis_bps=basis_bps)

    def safe_to_open(self, symbol: str, spot_tf: str = "5m") -> tuple[bool, str]:
        b = self.sample(symbol, spot_tf)
        if b is None:
            return False, "no basis sample yet"
        if abs(b.basis_bps) > self.entry_block_bps:
            return False, f"basis {b.basis_bps:+.0f}bps > entry block {self.entry_block_bps:.0f}bps"
        return True, "ok"

    def needs_exit_alert(self, symbol: str, spot_tf: str = "5m") -> Optional[BasisSample]:
        b = self.sample(symbol, spot_tf)
        if b is None:
            return None
        if abs(b.basis_bps) > self.exit_alert_bps:
            return b
        return None
