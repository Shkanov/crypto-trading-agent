"""Thin Anthropic SDK wrapper with prompt caching and tool-use loop helpers.

All LLM agents share this. Keeping the surface small makes it easy to swap or
mock for tests. Prompt caching is on by default for the system block; per-agent
caching of tool definitions is enabled where it pays off.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import structlog
from anthropic import AsyncAnthropic

from src.config.settings import get_settings

log = structlog.get_logger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any]  # async callable: (**kwargs) -> json-serializable


@dataclass
class LLMResult:
    text: str
    tool_calls_made: int = 0
    stopped: str = "end_turn"  # end_turn | max_tokens | tool_use_loop | budget_exceeded
    raw: list[dict] = field(default_factory=list)


# Per-model rough cost estimates ($ per 1M tokens, input + output blended).
# Update from current pricing periodically. Conservative: assume worst case.
_COST_PER_MTOK = {
    "claude-opus-4-7": 30.0,
    "claude-sonnet-4-6": 6.0,
    "claude-haiku-4-5-20251001": 1.5,
}


class TokenBudget:
    """Rolling 24h USD budget per agent. Records usage from `anthropic` usage
    objects and rejects calls when the projected cost exceeds the budget."""

    def __init__(self, daily_usd: float) -> None:
        self.daily_usd = daily_usd
        # list of (ts_seconds, usd_cost)
        self._spend: list[tuple[float, float]] = []

    def _trim(self) -> None:
        cutoff = time.time() - 86_400
        self._spend = [(t, v) for t, v in self._spend if t > cutoff]

    def spent_24h_usd(self) -> float:
        self._trim()
        return sum(v for _, v in self._spend)

    def remaining_usd(self) -> float:
        return max(0.0, self.daily_usd - self.spent_24h_usd())

    def can_spend(self, projected_usd: float) -> bool:
        return self.remaining_usd() >= projected_usd

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        rate = _COST_PER_MTOK.get(model, 5.0)
        cost = (input_tokens + output_tokens) / 1_000_000 * rate
        self._spend.append((time.time(), cost))
        return cost


class LLMAgent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: Optional[str] = None,
        tools: Optional[list[ToolSpec]] = None,
        max_tokens: int = 2048,
        max_tool_iters: int = 8,
        cache_tools: bool = True,
        budget: Optional[TokenBudget] = None,
    ) -> None:
        s = get_settings()
        self.name = name
        self.model = model or s.opus_model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.max_tokens = max_tokens
        self.max_tool_iters = max_tool_iters
        self.cache_tools = cache_tools
        self.budget = budget
        self._client = AsyncAnthropic(api_key=s.anthropic_api_key)

    def _tool_block(self) -> list[dict]:
        if not self.tools:
            return []
        blocks: list[dict] = []
        for t in self.tools:
            blocks.append({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            })
        # Cache control on the last tool only — that caches the whole tools block.
        if self.cache_tools and blocks:
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
        return blocks

    def _system_block(self) -> list[dict]:
        return [{
            "type": "text", "text": self.system_prompt,
            "cache_control": {"type": "ephemeral"},
        }]

    async def run(self, user_message: str) -> LLMResult:
        """Run the tool-use loop until the model stops or hits the iteration cap."""
        if self.budget and not self.budget.can_spend(0.001):
            log.warning("llm.budget_exceeded", name=self.name,
                        spent=self.budget.spent_24h_usd(),
                        cap=self.budget.daily_usd)
            return LLMResult(text="", stopped="budget_exceeded")
        messages: list[dict] = [{"role": "user", "content": user_message}]
        tool_calls = 0
        raw_dump: list[dict] = []
        for _ in range(self.max_tool_iters):
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._system_block(),
                tools=self._tool_block() if self.tools else None,
                messages=messages,
            )
            if self.budget and resp.usage:
                cost = self.budget.record(
                    self.model, resp.usage.input_tokens, resp.usage.output_tokens,
                )
                if not self.budget.can_spend(0.001):
                    log.warning("llm.budget_exhausted_mid_run", name=self.name,
                                last_cost=cost, total=self.budget.spent_24h_usd())
                    return LLMResult(text="", tool_calls_made=tool_calls,
                                     stopped="budget_exceeded", raw=raw_dump)
            raw_dump.append({"stop_reason": resp.stop_reason,
                             "usage": resp.usage.model_dump() if resp.usage else None})
            if resp.stop_reason != "tool_use":
                text_parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
                return LLMResult(
                    text="\n".join(text_parts), tool_calls_made=tool_calls,
                    stopped=resp.stop_reason, raw=raw_dump,
                )
            # Append the assistant turn (full content) and run any tool_use blocks.
            messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
            tool_results: list[dict] = []
            for block in resp.content:
                if getattr(block, "type", "") != "tool_use":
                    continue
                tool_calls += 1
                tool = next((t for t in self.tools if t.name == block.name), None)
                if not tool:
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": f"unknown tool {block.name}", "is_error": True,
                    })
                    continue
                try:
                    out = await tool.handler(**(block.input or {}))
                except Exception as e:
                    log.exception("tool.error", agent=self.name, tool=tool.name)
                    out = {"error": str(e)}
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": json.dumps(out, default=str)[:32_000],
                })
            messages.append({"role": "user", "content": tool_results})
        return LLMResult(text="", tool_calls_made=tool_calls, stopped="tool_use_loop", raw=raw_dump)
