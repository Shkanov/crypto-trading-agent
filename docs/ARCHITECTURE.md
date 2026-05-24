# Architecture

Intra-day crypto trading agent on Binance (spot + USDT-M perps), with LLM
reasoning for strategy/news/anomaly and deterministic code on the hot path.
Hybrid approval (small trades auto-execute, large trades require Telegram
confirmation). Built for local development first; containerized for VPS later.

## Core invariant

**LLMs produce policy, deterministic code enforces it.**

The LLM never sits on a per-tick path. It writes a versioned `StrategyConfig`
object (entry/exit rules, allowed symbols, sizing parameters, kill switches).
The hot loop reads that struct on each tick — no model call, just a lookup.

## Two-loop topology

```
HOT PATH (asyncio, target <50ms per tick)        COLD PATH (LLM, seconds)
────────────────────────────────────────         ──────────────────────────────
  Binance WebSocket (kline/trade/depth)            StrategyAgent       (Opus, every 5–15m)
    │                                              NewsSentimentAgent  (Haiku, every 1–2m)
    ▼                                              AnomalyInvestigator (Opus, on-trigger)
  IndicatorEngine (rolling buffers, TA-Lib)        TelegramAgent       (Haiku chat / Opus reasoning)
    │
    ▼
  SignalGenerator (rule-based confluence)          ─── writes ───►  StrategyConfig (versioned)
    │                                              ─── reads ────   SignalEvent / FillEvent / Anomaly streams
    ▼
  RiskGate (deterministic guardrails)
    │
    ▼
  Proposal → Approval (auto if small, Telegram if large) → Executor
    │
    ▼
  Binance REST (idempotent client_order_id, reconciled on boot)
```

Hot path → cold path: append-only event stream (signals, fills, anomalies).
Cold path → hot path: atomic `StrategyConfig` swap, gated by schema validation
and dry-run replay against the last N minutes of ticks.

## Subagent topology

| Agent                | Model            | Tools                                            | Cadence         |
|----------------------|------------------|--------------------------------------------------|-----------------|
| MarketDataMonitor    | none (Python)    | Binance WS                                       | continuous      |
| IndicatorEngine      | none             | (internal)                                       | per tick        |
| SignalGenerator      | none (rules)     | reads StrategyConfig                             | per tick        |
| RiskGate             | none             | deterministic gates                              | per proposal    |
| Executor             | none             | Binance REST                                     | per approved    |
| StrategyAgent        | Opus 4.7         | read signals, positions, news digest; write cfg  | every 5–15 min  |
| NewsSentimentAgent   | Haiku 4.5        | CryptoPanic, RSS, Whale Alert, DefiLlama         | every 1–2 min   |
| AnomalyInvestigator  | Opus 4.7         | tick history, news, on-chain                     | on trigger      |
| TelegramAgent        | Haiku / Opus     | send/receive, approval buttons, command parser   | event-driven    |
| Supervisor           | none             | restart, health checks, kill switch              | continuous      |

Prompt caching is mandatory on Haiku polling agents (system prompt + tool
defs + static context) — the dominant cost win.

## Indicators (starter set)

- EMA(21), EMA(55) — trend regime per timeframe
- MACD(12,26,9) — momentum confirmation (histogram zero-cross)
- RSI(14) — divergences and overbought/oversold context
- ATR(14) — volatility normalization for sizing and stops
- Bollinger Bands(20, 2σ) — squeeze + mean-reversion in range
- VWAP (session-anchored) — institutional reference, intra-day magnet
- Supertrend(10, 3) — discrete trend flips on 15m/1h
- CVD (from aggTrades) — order-flow confirmation for breakouts

Multi-timeframe rule: never trade against the 1h regime on a 1m signal.

## Signal scoring (Phase 1 — weighted vote)

```
score = 0.35·trend + 0.25·momentum + 0.20·volume + 0.10·volatility + 0.10·pattern
side  = long if score > 0.35 else short if score < -0.35 else flat
```

Each feature is normalized to [-1, 1]. Confidence is `|score|`. Phase 2 is a
gradient-boosted classifier on the same features with isotonic calibration;
Phase 3 is meta-labeling. Build Phase 1 first.

## Risk guardrails (defaults)

| Guardrail                          | Default        |
|------------------------------------|----------------|
| Risk per trade                     | 0.5–1% equity  |
| Max daily loss (kill switch)       | 3% → halt 24h  |
| Max weekly loss                    | 7% → halt 1wk  |
| Max concurrent positions           | 3–5            |
| Max exposure per coin              | 20% notional   |
| Max leverage (perps)               | 3–5x           |
| Drawdown trigger (size cut 50%)    | 15% peak-to-trough |
| Tilt: 3 consecutive losses         | halt 4h        |
| Tilt: 5 consecutive losses         | halt 24h       |
| Min gap between trades             | 5–15 min       |
| Min edge after costs               | edge > 2× (fees + slippage) |
| Liquidation distance (perps)       | ≥ 4× ATR(1h) from entry |

Correlation: total BTC-beta-weighted exposure capped at 1.5× single-position
size. Five long alts ≠ five independent bets.

## Approval flow

State machine in storage (SQLite local / Postgres prod):

```
PROPOSED
  │
  ├── (small)  → AUTO_APPROVED → SUBMITTED → FILLED/FAILED
  └── (large)  → AWAITING_USER → APPROVED/REJECTED/EXPIRED → SUBMITTED → FILLED/FAILED
```

- `large` = `notional > THRESHOLD_USD` OR `leverage > 3x` (configurable).
- Telegram message carries proposal ID, LLM-generated rationale, inline
  Approve / Reject / Modify buttons.
- Hard timeout (default 5 min). Expiry never auto-approves.
- Idempotency key = `proposal_id`. Exchange's `newClientOrderId` set to
  `cta_<proposal_id>` so retries dedupe.
- Optional 2FA: trades above `2FA_THRESHOLD` require a randomized 4-digit
  confirmation code in a follow-up Telegram reply, or a TOTP from an
  authenticator app.

## Failure modes & defenses

| Failure                              | Defense                                                        |
|--------------------------------------|----------------------------------------------------------------|
| Hallucinated ticker                  | LLM picks from `allowed_symbols` enum; executor revalidates against `exchangeInfo` |
| Double-spend on retry                | Deterministic `client_order_id` (UUIDv5 from proposal ID)      |
| LLM panic-unwind                     | StrategyAgent cannot close positions directly; mass-flatten is its own approval flow |
| Stale data fill                      | Reject orders whose triggering signal is older than N ms       |
| WebSocket gap                        | Snapshot orderbook on reconnect; halt hot path on gap          |
| Runaway losses                       | Deterministic daily-loss kill switch outside LLM control       |
| Exchange API outage                  | Circuit breaker on Executor; degrade to read-only              |
| LLM cost blowup                      | Per-agent token budget; Supervisor kills + restarts on overrun |
| Bad config deploy                    | Schema validation + dry-run replay before atomic swap          |
| Clock skew                           | NTP sync, time-offset from `GET /api/v3/time` on boot          |
| Symbol filter rejection              | `floor`-quantize to `LOT_SIZE.stepSize` / `PRICE_FILTER.tickSize` via `Decimal` |
| Unmatched state on boot              | Reconcile positions + open orders against exchange before hot loop starts |

## Data sources

- **Market data**: Binance WS — kline, trade, depth, mark price, force orders.
- **News firehose**: CryptoPanic free + CoinDesk / The Block / Decrypt RSS.
- **Macro sentiment**: alternative.me Fear & Greed (1/hour).
- **Flow / smart money**: Whale Alert webhooks; DefiLlama (stablecoin mints,
  bridge flows); Coinglass free (funding, OI, liquidations); Arkham free.
- **Ad-hoc deep-dive**: Perplexity Sonar (used by AnomalyInvestigator only).
- Skip for now: X API (priced out), Glassnode (wrong timeframe), Discord.

## Storage

- **Local dev**: SQLite via SQLAlchemy (single file, WAL mode).
- **Production**: Postgres in docker-compose; same SQLAlchemy code.
- **Hot state** (latest prices, rolling indicators, pending-approval queue):
  in-process Python dicts now; pluggable Redis backend later.

Boot reconciliation: read Postgres → call `GET /api/v3/account` and
`GET /fapi/v2/positionRisk` → if mismatch, halt + alert. Hot loop does not
start until reconciliation passes.

## Build order

1. Hot path skeleton (WS → indicators → rules → paper executor).
2. Storage + reconciliation on boot.
3. Approval state machine + Telegram bot (buttons only, no LLM yet).
4. StrategyAgent writing `StrategyConfig` with validation/dry-run gate.
5. NewsSentimentAgent with prompt caching.
6. AnomalyInvestigator + LLM-narrated risk reports.
7. Cut over from paper to live with tiny size limits; expand gradually.
