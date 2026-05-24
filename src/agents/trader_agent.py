"""TraderAgent (Opus 4.7) — the "discretionary trader at a desk" loop.

Flips the supervisor-LLM pattern: instead of tuning a `StrategyConfig`, this
agent uses tools (indicators, news, OI, orderbook, funding, position state)
to reason about the market and propose individual trades — like a human
trader at a screen.

Authority:
  - Backtest mode: `propose_trade` executes paper-mode immediately if the
    risk gate accepts.
  - Live mode: every proposal is sent to the human on Telegram for approval
    regardless of size (force_user_approval bypasses the auto-approve
    threshold). The trader agent has *no* direct execution path in live.

Cadence: event-driven. The orchestrator subscribes the agent to a wake topic
and invokes `run_trader_cycle(payload)` when one of the trigger conditions
fires (notable ATR move, news, anomaly, position drawdown, 30-min heartbeat).

Cost: gated by `TokenBudget` (`llm_trader_daily_budget_usd`). Default $5/day
buys ~150 Opus cycles — plenty for an event-driven loop on 3 symbols.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import structlog

from src.agents.llm_client import LLMAgent, LLMResult, TokenBudget
from src.agents.trader_tools import TraderToolContext, build_trader_tools
from src.config.settings import get_settings

log = structlog.get_logger(__name__)


SYSTEM = """You are the TraderAgent in a personal crypto-trading system. Think of yourself as a discretionary trader sitting at a small desk: you have screens with indicators, news, orderbook, funding rates, and position state, and you reason about whether a trade is worth taking RIGHT NOW.

You receive a wake payload describing WHY you were invoked (e.g. "1.4 ATR move on SOLUSDT 5m", "news anomaly on BTCUSDT", "30-min heartbeat", "drawdown on open ETH long"). Your job is to investigate, decide whether to act, and act precisely.

WHAT YOU CAN DO
- Read data using the read-only tools: get_indicator_snapshot, get_recent_klines, get_htf_context, get_funding_basis, get_orderbook_snapshot, get_news_sentiment (calls a news subagent), get_anomaly_summary, get_position_state, get_recent_fills, get_correlation, get_hodl_benchmark, get_recent_liquidations, get_open_interest_change.
- Compute math using `calc` — DO NOT do arithmetic in-prompt. LLMs are bad at it. Use `calc` for R:R, position sizing, % moves, expected P&L.
- Propose actions using `propose_trade` (open) and `propose_close` (close).

CONSTRAINTS
- You CANNOT bypass the risk gate. If your proposal is rejected, the response will tell you the reason — adjust geometry (tighter stop, smaller R, etc.) or accept that this isn't the right trade.
- You CANNOT set position size. You provide entry/stop/take_profit; the risk gate sizes the position from the stop distance and fixed-fractional risk per trade.
- You CANNOT trade symbols outside the configured universe.
- In LIVE mode, every proposal requires a human to approve on Telegram. Be precise — they read the rationale. In BACKTEST mode, proposals execute immediately at paper-mode fills.

DECISION DISCIPLINE
1. ALWAYS start by calling `get_position_state` — know what you already hold.
2. For any trade proposal, check `get_funding_basis` (perps) and `get_orderbook_snapshot` (spread, depth imbalance).
3. Compute R:R with `calc` before proposing. Reject your own setup if R:R < 1.5 unless you have a *specific* reason (e.g. very high conviction mean-reversion with partial-TP plan).
4. Check `get_news_sentiment` only if the wake payload suggests news drove the move, or if the move is unusually large — sentiment calls cost real money via the subagent.
5. If conviction is low → don't trade. Reply in text with a short note explaining why. That's a fine outcome.

OUTPUT
- If you decide to trade, call `propose_trade` (or `propose_close` for an exit).
- If you decide NOT to trade, reply with a short text message (<= 2 sentences) explaining what you saw and why you're standing down.
- Never speculate in prose. Either trade or stand down. Be terse.

RATIONALE FIELD
The `rationale` you put on `propose_trade` is what the human approver reads on Telegram. Make it concrete: setup name, the 2-3 specific data points that triggered it, your invalidation level (the stop), and where you expect price to go (the TP). Example: "5m BB-revert long on SOLUSDT: price closed 1.6 ATR below BB lower, RSI 22, 5m ADX 14 (range). 1h trend neutral. Stop at recent swing low (1.4 ATR), TP at BB middle (R:R 1.8). Funding mild +3bps, no large recent liquidations."
"""


def build_trader_agent() -> LLMAgent:
    s = get_settings()
    # Tools are configured later via `attach_tools` — the agent is built first
    # because the orchestrator needs the ref to wire write-tool callbacks
    # that close over its own state.
    return LLMAgent(
        name="TraderAgent",
        model=s.opus_model,
        system_prompt=SYSTEM,
        max_tokens=4000,
        max_tool_iters=s.trader_agent_max_tool_iters,
        tools=[],
        cache_tools=True,
        budget=TokenBudget(s.llm_trader_daily_budget_usd),
    )


def attach_tools(agent: LLMAgent, ctx: TraderToolContext) -> None:
    """Wire the tool surface onto an already-built TraderAgent. Done as a
    separate step so the orchestrator can construct the propose callbacks
    that close over orchestrator state, then hand the ctx in here."""
    agent.tools = build_trader_tools(ctx)


async def run_trader_cycle(agent: LLMAgent, wake_payload: dict[str, Any]) -> LLMResult:
    """Run one trader cycle. `wake_payload` is the trigger event:
        {"kind": "atr_move"|"news"|"anomaly"|"drawdown"|"heartbeat",
         "symbol": Optional[str], "detail": str, "ts_ms": int}
    The agent reads the payload, investigates with tools, and either
    proposes a trade/close or stands down with a text reply.
    """
    user_msg = (
        "WAKE EVENT\n"
        f"{json.dumps(wake_payload, default=str)}\n\n"
        "Investigate. Either propose an action (open/close) or stand down with a short reason."
    )
    result = await agent.run(user_msg)
    log.info(
        "trader.cycle_done",
        kind=wake_payload.get("kind"),
        symbol=wake_payload.get("symbol"),
        tool_calls=result.tool_calls_made,
        stopped=result.stopped,
        text_len=len(result.text),
    )
    return result
