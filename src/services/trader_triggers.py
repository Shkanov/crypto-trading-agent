"""Event-driven wake triggers for the TraderAgent.

Each public method evaluates a potential trigger and returns either a wake
payload (dict) or None. Per-kind cooldown deduplication prevents the agent
from being woken every bar when a long move is unfolding — once per
`min_wake_gap_sec` per kind. The orchestrator calls these from its existing
event hooks (closed-bar in `_stream_klines`, news in `_news_loop`,
anomaly in `_handle_anomaly`) and from a periodic heartbeat task.

Wake payload schema (kept compact — fed as the user message to the agent):
    {
        "kind": "atr_move" | "news" | "anomaly" | "position_drawdown" | "heartbeat",
        "symbol": str | None,
        "detail": str,           # one-line human-readable why
        "ts_ms": int,
    }
"""
from __future__ import annotations

import time
from typing import Any, Optional

import structlog

from src.config.settings import Settings, get_settings
from src.models.types import IndicatorSnapshot, Kline, Position

log = structlog.get_logger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


class WakeTriggers:
    """Stateless aside from per-kind cooldown timestamps and the last
    heartbeat time. Cheap to allocate, cheap to call on every bar."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.s = settings or get_settings()
        self._last_wake_ms: dict[str, int] = {}
        # Heartbeat starts fresh on construction — first heartbeat fires
        # after `trader_agent_heartbeat_sec` from init, not immediately.
        self._last_heartbeat_ms = _now_ms()

    # ---- shared cooldown check ---------------------------------------------

    def _cooldown_ok(self, kind: str, now_ms: int) -> bool:
        last = self._last_wake_ms.get(kind, 0)
        return (now_ms - last) >= self.s.trader_agent_min_wake_gap_sec * 1000

    def _mark(self, kind: str, now_ms: int) -> None:
        self._last_wake_ms[kind] = now_ms

    # ---- triggers -----------------------------------------------------------

    def on_closed_bar(self, k: Kline, snap: IndicatorSnapshot) -> Optional[dict[str, Any]]:
        """Wake on closed bar with |close - open| > N ATR. ATR is the
        indicator engine's atr14; if unavailable, no trigger fires."""
        if snap.atr14 is None or snap.atr14 <= 0:
            return None
        move = abs(k.close - k.open)
        atr_units = move / snap.atr14
        if atr_units < self.s.trader_agent_wake_atr_threshold:
            return None
        now = _now_ms()
        # Cooldown keyed per-symbol so an ETH move doesn't suppress a BTC move.
        kind = f"atr_move:{k.symbol}"
        if not self._cooldown_ok(kind, now):
            return None
        self._mark(kind, now)
        direction = "up" if k.close > k.open else "down"
        return {
            "kind": "atr_move",
            "symbol": k.symbol,
            "detail": (
                f"{atr_units:.2f}-ATR {direction} move on {k.symbol} {k.timeframe} "
                f"(open {k.open:.4f} -> close {k.close:.4f})"
            ),
            "ts_ms": k.close_time,
        }

    def on_news(self, news_item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Wake on news with sentiment magnitude above threshold. The
        `news_item` shape mirrors what `_news_loop` publishes — at minimum
        a `score` (-1..1) and `symbol`."""
        score = float(news_item.get("score", 0.0))
        if abs(score) < self.s.trader_agent_wake_news_sentiment_threshold:
            return None
        now = _now_ms()
        symbol = news_item.get("symbol") or "MARKET"
        kind = f"news:{symbol}"
        if not self._cooldown_ok(kind, now):
            return None
        self._mark(kind, now)
        return {
            "kind": "news",
            "symbol": symbol,
            "detail": (
                f"news sentiment {score:+.2f} on {symbol}: "
                f"{(news_item.get('summary') or '')[:160]}"
            ),
            "ts_ms": now,
        }

    def on_anomaly(self, anomaly_payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Wake on any anomaly of severity warn or critical."""
        sev = anomaly_payload.get("severity", "info")
        if sev not in ("warn", "warning", "critical"):
            return None
        now = _now_ms()
        symbol = anomaly_payload.get("symbol") or "MARKET"
        kind = f"anomaly:{symbol}:{anomaly_payload.get('kind', '?')}"
        if not self._cooldown_ok(kind, now):
            return None
        self._mark(kind, now)
        return {
            "kind": "anomaly",
            "symbol": symbol,
            "detail": (
                f"{sev} anomaly {anomaly_payload.get('kind')}: "
                f"{anomaly_payload.get('detail') or ''}"
            )[:300],
            "ts_ms": now,
        }

    def on_position_pressure(
        self, positions: list[Position], last_prices: dict[str, float],
    ) -> Optional[dict[str, Any]]:
        """Wake if any open position is in drawdown beyond the threshold
        (% of entry, not % of equity — the agent can reason about whether
        to manage the position)."""
        threshold = self.s.trader_agent_wake_position_drawdown_pct / 100.0
        worst_sym: Optional[str] = None
        worst_dd: float = 0.0
        for p in positions:
            px = last_prices.get(p.symbol)
            if not px or p.entry <= 0:
                continue
            if p.side == "long":
                dd = (p.entry - px) / p.entry
            else:
                dd = (px - p.entry) / p.entry
            if dd > worst_dd:
                worst_dd = dd
                worst_sym = p.symbol
        if worst_sym is None or worst_dd < threshold:
            return None
        now = _now_ms()
        kind = f"drawdown:{worst_sym}"
        if not self._cooldown_ok(kind, now):
            return None
        self._mark(kind, now)
        return {
            "kind": "position_drawdown",
            "symbol": worst_sym,
            "detail": f"open position on {worst_sym} is in {worst_dd:.2%} drawdown",
            "ts_ms": now,
        }

    def check_heartbeat(self) -> Optional[dict[str, Any]]:
        """Returns a wake payload if the heartbeat interval has elapsed
        since the last heartbeat. Caller invokes this periodically (e.g.
        every 60s); the trigger itself is rate-limited internally."""
        now = _now_ms()
        if (now - self._last_heartbeat_ms) < self.s.trader_agent_heartbeat_sec * 1000:
            return None
        self._last_heartbeat_ms = now
        return {
            "kind": "heartbeat",
            "symbol": None,
            "detail": "30-min routine check-in — nothing specific triggered this",
            "ts_ms": now,
        }
