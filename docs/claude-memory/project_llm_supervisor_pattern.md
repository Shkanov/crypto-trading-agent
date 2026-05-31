---
name: project-llm-supervisor-pattern
description: "The original LLM-agent pattern in this repo is \"LLM-as-supervisor\" — never places trades, only tunes configs / investigates anomalies. The new trader-agent flips this."
metadata: 
  node_type: memory
  type: project
  originSessionId: 6a6d183b-f5db-43e5-ae1c-04c3241d690c
---

Existing LLM agents under `src/agents/` follow a strict supervisor pattern:

- **StrategyAgent** (Opus 4.7): only output is a new `StrategyConfig` (weights, thresholds). System prompt explicitly forbids trades. Runs every ~10 min.
- **NewsAgent** (Haiku 4.5): outputs sentiment scores + market summary. No trades.
- **AnomalyInvestigator** (Opus 4.7): outputs an action enum {continue, pause, flatten}; orchestrator applies it via the same paths used for `/pause` and `/flatten` commands. Never closes positions directly.

All three share `LLMAgent` in `src/agents/llm_client.py` — tool-use loop, prompt caching, 24h USD `TokenBudget`.

**Why this matters:** the new TraderAgent flips this paradigm — it's an *operator*, not a supervisor. Trades flow Signal→Proposal→risk_gate→Telegram-approval(live)/direct-exec(backtest)→Executor. Keep the supervisor agents intact; the trader is an *additional* loop, not a replacement.

**How to apply:** when building new LLM-touching code in this repo, default to the supervisor pattern (LLM emits a structured config, deterministic code applies it) unless the work is specifically the trader-agent. The supervisor pattern was a deliberate safety choice — preserve it.

See also: [[project-trader-agent-design]].
