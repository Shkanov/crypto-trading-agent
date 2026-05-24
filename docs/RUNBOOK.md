# Runbook

What to check, when, and what to do on common alerts.

## Daily (2 min)

1. Read the daily digest in Telegram.
2. Confirm `/status`:
   - `Halted: no` (unless intended)
   - `vs BTC HODL` outperformance ≥ 0 over the trailing week
   - Funding rates on the symbols you trade are within `entry_threshold ± 5 bps`
3. If P&L today is meaningfully negative, look at the open positions —
   are they ones the strategy genuinely should be in given current funding?

## Weekly (15 min)

1. Look at the perf attribution in `/status`. Which strategy made money?
   Which symbol contributed most? Any symbols dragging?
2. Confirm the notional-ramp message landed (`Ramp up: …` or `Ramp hold: …`).
   If ramp halved, dig into the loss day.
3. Compare cumulative P&L to BTC HODL. If 4 weeks in a row of
   underperformance, the agent auto-halts; you'll see a CRITICAL Telegram.

## Monthly

1. Run the backtest on fresh data:
   ```bash
   BINANCE_TESTNET=false uv run python -m scripts.backtest \
     --strategy funding --symbol BTCUSDT --days 60
   ```
   Deflated Sharpe of the proven strategy should hold > 0.5. Below that for
   two months, halt and investigate regime change.
2. Rotate the Binance API key. Update `.env`, redeploy, verify reconciliation
   passes on boot.
3. Test the DB restore (scratch VPS) — see DEPLOY.md §10.

## Common alerts

### `BOOT RECONCILIATION FAILED`

The agent has halted itself for 24h. Local DB and exchange state disagree.

1. Read the report in Telegram. Two cases:
   - **Ghost** (local-only): a Trade row says there's a position on the
     exchange that isn't there. Probably you (or someone) closed it
     manually on Binance. SSH in, run a SQL update to set the Trade to
     `status=CLOSED` with `exit_reason=manual_external`, then resume.
   - **Orphan** (exchange-only): there's a Binance position the agent
     doesn't know about. Either you opened a manual trade, or a previous
     run lost state. Either close it on Binance, or insert a matching
     Trade row before resume.
2. After resolving, remove the kill-switch file if present and `/resume`
   on Telegram.

### `CLOCK SKEW Xms exceeds tolerance`

Host NTP isn't keeping up. Binance rejects orders with `recvWindow` issues.

```bash
sudo systemctl status chronyd     # check service is running
sudo chronyc tracking             # see current offset
sudo chronyc makestep             # force immediate correction
```

If chrony is healthy and skew persists, your VPS provider may be using a
flaky clock source. Switch NTP servers or escalate.

### `Daily loss cap hit`

Trading auto-halts for 24h. **Don't bypass this.** Sleep on it. Tomorrow,
review:

1. Which strategy lost money? Pull `recent_closed_trades` from storage.
2. Was the loss in line with what the strategy is supposed to lose
   (i.e., a stop got hit cleanly), or was it a fat-fingered config / bug?
3. If config issue, push the fix, deploy. If just a bad day, let the halt
   expire and let the ramp policy halve `max_notional_usd`. Don't override.

### `KILL SWITCH file present`

Either you put it there, or someone with SSH access did. Find out which
before doing anything else. If it was you and you're done, delete it.
Otherwise: rotate SSH keys, audit `last -F`, audit Binance API key access
logs.

### LLM agent stops responding

Likely the token budget hit zero. Check Telegram for the `llm.budget_exceeded`
log entry. The agent will resume when the rolling 24h window clears the spend.
If you need it back immediately, raise the budget in `.env` and restart.

### `agent_ws_reconnects_total` climbing

WS instability. Usually the upstream Binance side; sometimes the VPS network.

1. Check `binance.com` status page.
2. `ping fapi.binance.com` from the VPS.
3. If many reconnects but the agent is processing klines OK, it's fine —
   the supervisor handles it. If reconnects are racing and indicators aren't
   updating, restart the agent.

## Manual operations

### Force-close a single position

Closing via the Binance UI works but the reconciliation check will halt on
next boot. Prefer:

```
# In Telegram:
/flatten        # closes EVERYTHING and halts 1h
```

A single-trade close currently requires a SQL update + manual exchange close.
A `/close <trade_id>` command is on the roadmap.

### Roll back to a previous version

```bash
ssh agent@vps
cd /opt/crypto-trading-agent
git checkout vX.Y.Z-1   # previous tag
sudo systemctl restart crypto-trading-agent
```

If the DB schema changed between versions, you'll need to restore from the
backup taken right before the upgrade. This is why we backup before each
upgrade.

### Sanity-check the agent is doing what you think

```bash
# What trades has it taken today?
docker compose exec postgres psql -U agent -d agent \
  -c "SELECT id, strategy, symbol, side, qty, entry_price, exit_price, exit_reason, realized_pnl_usd FROM trades WHERE created_at > now() - interval '1 day' ORDER BY created_at DESC;"

# Open positions?
docker compose exec postgres psql -U agent -d agent \
  -c "SELECT id, strategy, symbol, side, qty, entry_price, funding_accrued_usd FROM trades WHERE status='OPEN';"

# What does the LLM strategy advisor want to change?
docker compose exec postgres psql -U agent -d agent \
  -c "SELECT id, base_version, notes, created_at FROM proposed_configs WHERE status='PENDING' ORDER BY created_at DESC LIMIT 5;"
```

## What this runbook deliberately does NOT cover

- **Profit-taking psychology** — that's on you.
- **When to size up** — the notional ramp handles it deterministically; don't
  override it on emotion.
- **When to add a new strategy** — only after the existing strategy has run
  profitably for 90+ days. Adding strategies that are themselves
  unvalidated is the fast track to fooling yourself.
