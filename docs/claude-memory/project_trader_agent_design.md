---
name: project-trader-agent-design
description: "Design decisions for the new TraderAgent (LLM-as-operator) — live uses Telegram approval, backtest uses direct paper execution; event-driven cadence; open+close authority."
metadata: 
  node_type: memory
  type: project
  originSessionId: 6a6d183b-f5db-43e5-ae1c-04c3241d690c
---

New regime added 2026-05-24: a TraderAgent that acts like a human trader at a desk — uses tools, calls subagents, reasons, then proposes trades. Lives alongside (does not replace) the existing supervisor agents [[project-llm-supervisor-pattern]].

**Authority (user-specified):**
- Backtest mode: agent's `propose_trade` calls execute directly via paper `Executor` (no human in loop).
- Live mode: every trade proposal — regardless of size — requires human approval on Telegram. Implementation: add `force_user_approval: bool` to `orchestrator._propose()` to bypass the `auto_approve_max_notional_usd` branch for trader-agent proposals only.

**Authority also includes closing positions:** agent gets `propose_close(position_id, rationale)` write tool. PositionManager still handles deterministic stop/TP/time-stop; the agent's close authority is *additive*, not exclusive.

**Cadence:** event-driven. Wake triggers:
1. Closed bar with `|close-open| > 1 ATR` on any allowed symbol
2. News item with sentiment magnitude > threshold
3. `TOPIC_ANOMALY` of severity ≥ warn
4. Open position drawdown > 0.5%
5. 30-min heartbeat (idle check-in)

Each trigger publishes to a new `TOPIC_TRADER_WAKE`. Min-gap dedup: 60s per trigger kind.

**Tool surface (user-confirmed):** indicator snapshots, recent klines, HTF context, funding/basis, orderbook snapshot, news sentiment (calls NewsAgent as subagent), anomaly summary, position state, recent fills, correlation, HODL benchmark, **recent liquidations**, **OI change**, **sandbox calculator** (LLMs are bad at arithmetic — give them a `calc(expr)` tool instead of expecting in-prompt math). Two write tools: `propose_trade(...)`, `propose_close(position_id, rationale)`.

**Why:** user wanted to mirror how a human trader actually works — has data on screen, reads news, reasons, then acts. The supervisor pattern alone can't capture discretionary trading style.

**How to apply:** the trader agent is a *parallel* operator loop, not a replacement for the rule-based strategies. Keep mean_reversion / indicator_confluence / funding_harvest running; the trader-agent is an additional source of Proposals into the same risk-gate funnel.

**Build order:** Settings fields → trader_tools.py (read handlers) → trader_agent.py (LLM + write tools) → trader_triggers.py (event-bus subscriber) → `_propose` edit → orchestrator wiring.

**In flight before pivot:** v2 scaled mean-reversion backtest (uncommitted) in `src/services/backtest.py` + `src/strategies/mean_reversion.py` + `scripts/backtest_grid.py`. Code parses + imports OK but unrun. Don't lose it.
