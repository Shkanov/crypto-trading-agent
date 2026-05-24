"""AnomalyInvestigator (Opus 4.7).

Fired ON-DEMAND when a deterministic detector trips (WS gap, sudden price
move > N ATR, liquidation cluster, funding spike, news-sentiment / price-action
divergence). Its job is to produce a human-readable diagnosis and a
RECOMMENDATION ENUM the orchestrator can act on:

  - "continue": looks normal, keep trading
  - "pause": halt new entries for N minutes, hold existing positions
  - "flatten": close all positions, halt 1h, request human review

The orchestrator enforces the action — the agent only recommends.
"""
from __future__ import annotations

import json
from typing import Literal

import structlog

from src.agents.llm_client import LLMAgent, TokenBudget
from src.config.settings import get_settings

log = structlog.get_logger(__name__)

SYSTEM = """You are the AnomalyInvestigator in a crypto trading system.

You are invoked when a deterministic detector trips. Your job:
1. Examine the anomaly + recent context (ticks, news, funding, position state).
2. Diagnose what's likely happening in <= 200 chars.
3. Recommend ONE action: "continue" | "pause" | "flatten".
4. Set severity: "info" | "warn" | "critical".
5. Output ONLY JSON: {"action": "...", "severity": "...", "diagnosis": "..."}.

Be CONSERVATIVE. "flatten" is for real disasters (exchange outage, halt rumor,
catastrophic news, our own data integrity loss). "pause" is for elevated risk
(big news pending, funding extremes, wild volatility). "continue" is the default.
"""


def build_anomaly_agent() -> LLMAgent:
    s = get_settings()
    return LLMAgent(
        name="AnomalyInvestigator",
        model=s.opus_model,
        system_prompt=SYSTEM,
        max_tokens=600,
        tools=[],
        cache_tools=False,
        budget=TokenBudget(s.llm_anomaly_daily_budget_usd),
    )


Action = Literal["continue", "pause", "flatten"]


_VALID_ACTIONS = {"continue", "pause", "flatten"}
_VALID_SEVERITIES = {"info", "warn", "critical"}


async def investigate(agent: LLMAgent, context: dict) -> tuple[Action, str, str]:
    user = json.dumps(context, default=str)
    result = await agent.run(user)
    try:
        parsed = json.loads(result.text)
    except json.JSONDecodeError:
        return "pause", "warn", "anomaly agent returned non-JSON; pausing as a precaution"
    # R8: normalize to lowercase + whitelist. LLMs sometimes return "FLATTEN"
    # or "close_all"; we default to the safest non-no-op action (pause) on
    # any unknown value rather than silently treating it as "continue".
    raw_action = str(parsed.get("action", "pause")).strip().lower()
    action: Action = raw_action if raw_action in _VALID_ACTIONS else "pause"
    raw_sev = str(parsed.get("severity", "warn")).strip().lower()
    severity = raw_sev if raw_sev in _VALID_SEVERITIES else "warn"
    return action, severity, parsed.get("diagnosis", "")
