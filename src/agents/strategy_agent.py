"""StrategyAgent (Opus 4.7).

Runs every ~10 min on a snapshot of:
- recent indicator readings per (symbol, TF)
- recent fills + P&L
- current sentiment scores + market summary
- current StrategyConfig

It proposes a NEW `StrategyConfig` (weights, thresholds, allowed sides,
allowed symbols) within tight bounds. Output is validated against the schema
AND replayed against the last N minutes of ticks (dry-run) before being
atomically swapped in. The hot loop never sees an unvalidated config.

The LLM cannot place trades, close positions, or bypass the risk gate.
Its only effect on the world is via `StrategyConfig`.
"""
from __future__ import annotations

import json
from typing import Optional

import structlog

from src.agents.llm_client import LLMAgent, TokenBudget
from src.config.settings import get_settings
from src.models.types import StrategyConfig

log = structlog.get_logger(__name__)

SYSTEM = """You are the StrategyAgent in a crypto trading system.

Your ONLY output is a new StrategyConfig (JSON). You CANNOT place trades, close
positions, or change risk limits. Your config is validated and dry-run before
being applied; a bad config is silently rejected.

You receive every cycle:
- The CURRENT StrategyConfig (you may keep it unchanged)
- A snapshot of indicators per (symbol, timeframe)
- Recent trades + P&L (last 24h)
- Sentiment scores per symbol + market summary
- Account stats (equity, open positions, drawdown)

Rules:
1. Output VALID JSON only, matching the StrategyConfig schema below. No prose.
2. allowed_symbols MUST be a subset of {allowed_symbols_universe}.
3. feature_weights values must be in [0, 0.6] and sum to ~1.0 (renormalized if not).
4. long_score_threshold in [0.20, 0.65]; short_score_threshold in [-0.65, -0.20].
5. min_confidence in [0.30, 0.75].
6. atr_stop_mult in [1.2, 3.5]. rr_target in [1.2, 3.0].
7. htf_regime_filter: prefer true unless you have STRONG reason to disable.
8. If recent P&L < -1% equity, REDUCE risk (raise thresholds, narrow allowed_sides).
9. If recent P&L > +2% equity AND > 60% win-rate, you MAY relax thresholds modestly.
10. Never disable risk filters by widening thresholds beyond the bounds above.
11. Always set `version = current_version + 1`.
12. Include a `notes` field (<=300 chars) explaining the change in plain English.

Schema:
{
  "version": int, "allowed_symbols": [str], "enabled_sides": ["long","short"],
  "feature_weights": {"trend": f, "momentum": f, "volume": f, "volatility": f, "pattern": f},
  "long_score_threshold": f, "short_score_threshold": f, "min_confidence": f,
  "atr_stop_mult": f, "rr_target": f, "htf_regime_filter": bool,
  "htf_timeframe": "1h"|"4h", "notes": str
}
"""


def build_strategy_agent() -> LLMAgent:
    s = get_settings()
    return LLMAgent(
        name="StrategyAgent",
        model=s.opus_model,
        system_prompt=SYSTEM,
        max_tokens=2000,
        tools=[],
        cache_tools=False,
        budget=TokenBudget(s.llm_strategy_daily_budget_usd),
    )


def validate_and_clamp(proposed: dict, current: StrategyConfig,
                       universe: list[str]) -> Optional[StrategyConfig]:
    try:
        # Subset check
        proposed["allowed_symbols"] = [s for s in proposed.get("allowed_symbols", []) if s in universe]
        if not proposed["allowed_symbols"]:
            return None

        # Weight bounds + renormalize
        fw = proposed.get("feature_weights", current.feature_weights)
        for k in ("trend", "momentum", "volume", "volatility", "pattern"):
            fw.setdefault(k, current.feature_weights[k])
            fw[k] = max(0.0, min(0.6, float(fw[k])))
        total = sum(fw.values()) or 1.0
        for k in fw:
            fw[k] = fw[k] / total
        proposed["feature_weights"] = fw

        proposed["long_score_threshold"] = max(0.20, min(0.65, float(proposed.get("long_score_threshold", current.long_score_threshold))))
        proposed["short_score_threshold"] = max(-0.65, min(-0.20, float(proposed.get("short_score_threshold", current.short_score_threshold))))
        proposed["min_confidence"] = max(0.30, min(0.75, float(proposed.get("min_confidence", current.min_confidence))))
        proposed["atr_stop_mult"] = max(1.2, min(3.5, float(proposed.get("atr_stop_mult", current.atr_stop_mult))))
        proposed["rr_target"] = max(1.2, min(3.0, float(proposed.get("rr_target", current.rr_target))))
        proposed["htf_regime_filter"] = bool(proposed.get("htf_regime_filter", current.htf_regime_filter))
        proposed["htf_timeframe"] = proposed.get("htf_timeframe", current.htf_timeframe)
        proposed["enabled_sides"] = [s for s in proposed.get("enabled_sides", current.enabled_sides) if s in ("long", "short")]
        if not proposed["enabled_sides"]:
            proposed["enabled_sides"] = list(current.enabled_sides)
        proposed["version"] = current.version + 1
        proposed["notes"] = str(proposed.get("notes", ""))[:300]

        return StrategyConfig.model_validate(proposed)
    except Exception as e:
        log.warning("strategy.validate_failed", err=str(e))
        return None


async def run_strategy_cycle(
    agent: LLMAgent,
    current: StrategyConfig,
    universe: list[str],
    snapshot: dict,
) -> Optional[StrategyConfig]:
    """`snapshot` is a free-form dict the orchestrator assembles each cycle."""
    user = json.dumps({
        "allowed_symbols_universe": universe,
        "current_config": current.model_dump(),
        "snapshot": snapshot,
    }, default=str)
    result = await agent.run(user)
    try:
        proposed = json.loads(result.text)
    except json.JSONDecodeError:
        log.warning("strategy.bad_json", text=result.text[:300])
        return None
    return validate_and_clamp(proposed, current, universe)
