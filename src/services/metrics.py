"""Prometheus metrics — exposed via a separate HTTP server on port 9090.

We expose gauges + counters that capture the operational state most likely
to need alerting on:

  - equity_usd, pnl_today_usd (operator-facing)
  - open_trades_total (sanity)
  - funding_rate_bps{symbol=...} (the alpha source for funding strategy)
  - basis_bps{symbol=...} (safety dimension)
  - ws_reconnects_total (data health)
  - llm_tokens_total{agent=...} + llm_budget_remaining_usd{agent=...}
  - position_unrealized_pnl_usd{symbol=...}

Reads are pulled by Prometheus scrape; we update gauges from the
housekeeping loop so the cost is bounded.
"""
from __future__ import annotations

from typing import Optional

import structlog
from prometheus_client import Counter, Gauge, start_http_server

log = structlog.get_logger(__name__)


equity_g = Gauge("agent_equity_usd", "Current equity (start + pnl_today)")
pnl_today_g = Gauge("agent_pnl_today_usd", "Realized P&L since UTC midnight")
hodl_outperf_g = Gauge("agent_hodl_outperformance_usd",
                       "Equity − HODL benchmark equity")
open_trades_g = Gauge("agent_open_trades_total", "Open Trade rows", ["strategy"])
pending_proposals_g = Gauge("agent_pending_proposals_total",
                            "Proposals in AWAITING_USER/AUTO_APPROVED")
halted_g = Gauge("agent_halted", "1 if trading halted, 0 otherwise")
consecutive_losses_g = Gauge("agent_consecutive_losses",
                             "Recent consecutive losing trades")

# Sprint #12 circuit breakers — three independent gauges so each can be
# alerted on separately.
circuit_size_mult_g = Gauge(
    "agent_circuit_size_multiplier",
    "Risk-circuit size multiplier applied to new positions (1.0, 0.5, or 0.0)",
)
circuit_dd_from_peak_g = Gauge(
    "agent_circuit_dd_from_peak_pct",
    "Trailing drawdown from rolling-peak equity, percent",
)
circuit_cooloff_active_g = Gauge(
    "agent_circuit_cooloff_active",
    "1 if a flatten-cooloff window is still in effect, 0 otherwise",
)

funding_rate_g = Gauge("agent_funding_rate_bps",
                       "Current 8h funding rate", ["symbol"])
basis_g = Gauge("agent_basis_bps", "Current (perp - spot) / spot", ["symbol"])
last_price_g = Gauge("agent_last_price", "Last seen kline close", ["symbol"])

ws_reconnects_c = Counter("agent_ws_reconnects_total",
                          "WS stream reconnect events", ["stream"])
anomaly_c = Counter("agent_anomalies_total", "Anomalies fired", ["kind", "severity"])
trade_open_c = Counter("agent_trades_opened_total", "Trades opened", ["strategy", "side"])
trade_close_c = Counter("agent_trades_closed_total",
                        "Trades closed", ["strategy", "exit_reason"])

llm_tokens_c = Counter("agent_llm_tokens_total",
                       "Cumulative tokens consumed", ["agent", "kind"])
llm_budget_g = Gauge("agent_llm_budget_remaining_usd", "Remaining USD budget",
                     ["agent"])


_server_started = False


def start_metrics_server(port: int = 9090) -> None:
    """Idempotent — safe to call multiple times. Binds to 0.0.0.0 inside the
    container; docker-compose maps it to 127.0.0.1:9090 for safety."""
    global _server_started
    if _server_started:
        return
    try:
        start_http_server(port)
        _server_started = True
        log.info("metrics.server_started", port=port)
    except OSError as e:
        log.warning("metrics.start_failed", port=port, err=str(e))
