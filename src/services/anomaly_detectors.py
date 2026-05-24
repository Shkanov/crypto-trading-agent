"""Deterministic anomaly detectors.

These fire BEFORE the LLM is consulted — cheap rule-based triggers that
identify "something unusual just happened, worth a closer look." The LLM
then investigates and recommends an action. The orchestrator enforces the
action.

Detectors are deliberately stateful so they don't fire on every bar in a
sustained anomaly — each has a cooldown and a threshold escalator.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import structlog

from src.models.types import Anomaly, IndicatorSnapshot, Kline, now_ms

log = structlog.get_logger(__name__)


@dataclass
class DetectorState:
    last_fire_ms: int = 0
    last_value: Optional[float] = None


class AnomalyDetectors:
    """Bundle of stateless rule-based detectors. Each call returns 0..N
    Anomaly instances; the orchestrator decides whether to invoke the LLM."""

    def __init__(self, cooldown_sec: int = 300) -> None:
        self.cooldown_ms = cooldown_sec * 1000
        # keyed by (kind, symbol)
        self._state: dict[tuple[str, str], DetectorState] = {}

    def _ok_to_fire(self, kind: str, symbol: str) -> bool:
        st = self._state.setdefault((kind, symbol), DetectorState())
        now = now_ms()
        if now - st.last_fire_ms < self.cooldown_ms:
            return False
        st.last_fire_ms = now
        return True

    # -------- Detectors --------

    def price_jump(self, k: Kline, snap: IndicatorSnapshot,
                   atr_multiplier: float = 5.0) -> Optional[Anomaly]:
        """Single-bar move > N * ATR. Captures news-driven gaps + flash crashes."""
        if snap.atr14 is None or snap.atr14 <= 0:
            return None
        move = abs(k.close - k.open)
        if move <= atr_multiplier * snap.atr14:
            return None
        if not self._ok_to_fire("price_jump", k.symbol):
            return None
        direction = "up" if k.close > k.open else "down"
        return Anomaly(
            symbol=k.symbol, kind="price_jump",
            detail=f"{direction} {move:.2f} ({move / snap.atr14:.1f}× ATR) on {k.timeframe}",
            severity="warn",
        )

    def funding_extreme(self, symbol: str, current_bps: Optional[float],
                        avg_bps: Optional[float],
                        threshold_bps: float = 50.0) -> Optional[Anomaly]:
        """Funding deviates dramatically — usually pre/post-news spike."""
        if current_bps is None:
            return None
        if abs(current_bps) < threshold_bps:
            return None
        if not self._ok_to_fire("funding_extreme", symbol):
            return None
        direction = "longs paying" if current_bps > 0 else "shorts paying"
        avg_part = f" (21-avg {avg_bps:+.1f}bps)" if avg_bps is not None else ""
        return Anomaly(
            symbol=symbol, kind="funding_extreme",
            detail=f"{current_bps:+.1f}bps {direction}{avg_part}",
            severity="warn" if abs(current_bps) < 100 else "critical",
        )

    def basis_blowout(self, symbol: str, basis_bps: Optional[float],
                      threshold_bps: float = 150.0) -> Optional[Anomaly]:
        """Perp/spot basis dislocated — directional risk creeps into 'neutral' pairs."""
        if basis_bps is None or abs(basis_bps) < threshold_bps:
            return None
        if not self._ok_to_fire("basis_blowout", symbol):
            return None
        return Anomaly(
            symbol=symbol, kind="basis_blowout",
            detail=f"basis {basis_bps:+.1f}bps (perp dislocated)",
            severity="critical" if abs(basis_bps) > 300 else "warn",
        )

    def ws_gap(self, symbol: str, interval: str) -> Optional[Anomaly]:
        """Fired explicitly by the orchestrator after a WS reconnect."""
        if not self._ok_to_fire(f"ws_gap_{interval}", symbol):
            return None
        return Anomaly(
            symbol=symbol, kind="ws_gap",
            detail=f"stream {interval} reconnected — possible missed ticks",
            severity="warn",
        )
