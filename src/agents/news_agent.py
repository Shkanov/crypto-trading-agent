"""News + sentiment agent (Haiku 4.5).

Pulls the latest items + macro readings every ~120s, asks Haiku to:
  1. Deduplicate by story cluster
  2. Score each cluster on a fixed rubric (catalyst type, direction, confidence)
  3. Emit per-symbol SentimentScore objects

The output is consumed by the StrategyAgent and surfaced to Telegram digests.
LLM never names tickers freely — it must pick from `allowed_symbols`.
"""
from __future__ import annotations

import json
from typing import Optional

import structlog

from src.agents.llm_client import LLMAgent, TokenBudget, ToolSpec
from src.config.settings import get_settings
from src.models.types import SentimentScore
from src.services.news import NewsService

log = structlog.get_logger(__name__)

SYSTEM = """You are the NewsSentimentAgent in a crypto trading system.

Your job: read recent crypto news + macro readings and emit a STRUCTURED sentiment
update for the assets the trader is watching. You do NOT pick tickers freely —
you only score the symbols in `allowed_symbols`.

Output ONLY valid JSON of shape:
{
  "scores": [
    {"symbol": "BTCUSDT", "score": -1..+1, "catalyst": "hack|listing|regulation|partnership|macro|flow|other",
     "confidence": 0..1, "rationale": "<=200 chars", "sources": ["...","..."]}
  ],
  "summary": "<=300 chars — one-line market state for the trader"
}

Rules:
- Dedupe story clusters: if 50 outlets repost the same hack, that's ONE signal.
- Sources tier: Bloomberg/CoinDesk/The Block > Decrypt/Cointelegraph > random Substack > anon X.
- Time-decay: anything > 60 min old contributes 0.5x; > 4 h old contributes 0.2x.
- If sentiment contradicts the symbol's recent price action (provided below),
  REDUCE confidence — don't bet against the tape on words alone.
- Macro context (fear/greed, stablecoin mcap) shifts ALL scores by ±0.2 max.
- If nothing material, return scores: [] and "summary": "no material news".
- Never invent tickers. Never wrap JSON in code fences.
"""


def build_news_agent(news: NewsService) -> LLMAgent:
    s = get_settings()
    return LLMAgent(
        name="NewsSentimentAgent",
        model=s.haiku_model,
        system_prompt=SYSTEM,
        max_tokens=1500,
        cache_tools=True,
        tools=[],
        budget=TokenBudget(s.llm_news_daily_budget_usd),
    )


async def run_news_cycle(
    agent: LLMAgent,
    news: NewsService,
    allowed_symbols: list[str],
    recent_returns_pct: dict[str, float],
) -> tuple[list[SentimentScore], str]:
    payload = await news.poll_once()
    items = payload["news"][:30]
    user = json.dumps({
        "allowed_symbols": allowed_symbols,
        "fear_greed": payload["fear_greed"],
        "fear_greed_classification": payload["fear_greed_classification"],
        "stablecoin_marketcap_usd": payload["stablecoin_marketcap_usd"],
        "recent_returns_pct_60min": recent_returns_pct,
        "news": [
            {"title": i.title, "source": i.source, "url": i.url, "coins": i.coins}
            for i in items
        ],
    }, default=str)

    result = await agent.run(user)
    try:
        parsed = json.loads(result.text)
    except json.JSONDecodeError:
        log.warning("news_agent.bad_json", text=result.text[:300])
        return [], ""

    scores: list[SentimentScore] = []
    for row in parsed.get("scores", []):
        sym = row.get("symbol")
        if sym not in allowed_symbols:
            continue
        try:
            scores.append(SentimentScore(
                symbol=sym,
                score=float(row.get("score", 0.0)),
                catalyst=row.get("catalyst", "other"),
                confidence=float(row.get("confidence", 0.0)),
                sources=row.get("sources", [])[:5],
            ))
        except (ValueError, TypeError):
            continue
    return scores, parsed.get("summary", "")
