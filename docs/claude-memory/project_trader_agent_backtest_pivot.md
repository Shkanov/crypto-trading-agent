---
name: project-trader-agent-backtest-pivot
description: TraderAgent backtest harness completed pivot from Anthropic SDK to Claude Code CLI subprocess + stdio MCP server. End-to-end smoke green on 2026-05-25.
metadata: 
  node_type: memory
  type: project
  originSessionId: ffba0bff-3baa-4529-84d1-a8b382028a8d
---

**Status:** PIVOT COMPLETE. End-to-end smoke green on 2026-05-25.

**Architecture (live in main):**
- Driver: `scripts/backtest_trader_agent.py`
- Harness: `src/services/backtest_trader.py` — owns per-run tempdir, writes snapshot.json before each wake, spawns `claude -p`, drains outbox.json after.
- MCP server: `src/agents/trader_mcp_server.py` — 16 tools; read tools load snapshot.json (env `TRADER_STATE_PATH`), write tools (propose_trade, propose_close) append to outbox.json (env `TRADER_OUTBOX_PATH`).
- Subprocess invocation: `claude -p --no-session-persistence --strict-mcp-config --mcp-config <tempdir>/mcp_config.json --model opus --max-budget-usd 0.50 --output-format json --tools "" --allowedTools mcp__trader-backtest --dangerously-skip-permissions --system-prompt <TRADER_SYSTEM_PROMPT> "<wake payload>"`

**Gotchas locked in (do not relearn):**
- DO NOT use `--bare`. Strictly bypasses keychain/OAuth, requires `ANTHROPIC_API_KEY` env. User has Claude.ai OAuth only.
- `--allowedTools mcp__trader-backtest` (no `__*` suffix, no wildcard) allows ALL tools from that MCP server. Wildcards are unreliable.
- mcp Python SDK installed via `uv pip install mcp --python .venv/bin/python` (the venv has a broken shebang for the bundled `pip` wrapper — use `uv pip` or `python -m pip`).
- MCP server command in mcp_config.json must be `sys.executable` (the venv python) so `mcp` module + editable `src` install both resolve.
- Result JSON has `total_cost_usd` (float) and `num_turns` (int) — that's how we track cost & turns instead of the old Anthropic SDK budget tracker.

**Smoke baseline (for regression detection):**
- BTCUSDT 5m 200 bars --max-wakes 2 → 1 wake fired (atr_move), 3 turns, $0.0461, 0 trades. Cost guard intact.

**What changed vs the pre-pivot harness (now gone):**
- Dropped `LLMAgent` / `build_trader_agent` / `attach_trader_tools` / `run_trader_cycle` imports
- Dropped in-process tool overrides (`_install_backtest_overrides`, `_stub_orderbook`, `_historical_funding_basis`, `_historical_oi_change`, `_historical_klines`, `_stub_news`, `_build_context`) — all live in MCP server now reading the snapshot
- Dropped the `anthropic_api_key` precheck — replaced with `shutil.which("claude")` precheck
- `TraderBacktestResult.total_tool_calls` renamed to `total_turns` (sourced from claude json output)

**Known cosmetic issue (pre-existing, not caused by pivot):**
- HTF indicator warmup processes ALL fetched HTF klines before replay starts, leaking future HTF state into pre-replay snapshots. Was true under the old harness too; deferred fix.

**Cursor-clock fix (2026-05-25):** WakeTriggers' per-kind cooldowns used wall-clock `_now_ms()`. In a fast replay the entire run finishes within seconds of wall time, so any cooldown >0s suppresses every wake after the first. Now WakeTriggers takes `clock_ms: Callable[[], int]`; harness passes `lambda: self.cursor_ts_ms`. Without this the agent only ever wakes once per backtest, no matter how many bars.

**Why: ** observed in the first 1500-bar/max-wakes=10 run — only 1 wake fired across 1300 replay bars until the clock was injected. After fix: 10 wakes / 2 proposals / 1 executed / 1 stop-hit closed trade / $0.32 total cost.

**How to apply:** any future trigger module that uses wall-clock for rate-limiting needs the same clock-injection treatment for backtest fidelity.

See also: [[project-trader-agent-design]] for the original live design, [[project-llm-supervisor-pattern]] for the surrounding agent ecosystem.
