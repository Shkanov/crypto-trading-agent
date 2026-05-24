# crypto-trading-agent

LLM-supervised intra-day crypto trading agent on Binance (spot + USDT-M perps).
Deterministic hot path for tick processing, Claude (Opus 4.7 / Haiku 4.5) for
strategy reasoning, news synthesis, and anomaly investigation. Telegram bot
for signal delivery, with hybrid approval (small trades auto-execute, large
trades wait for your tap).

> **This is research / personal-use software. Real money loss is possible.
> Start on Binance Testnet, then live with tiny size. Read `docs/ARCHITECTURE.md`
> before running.**

## Quick start (paper trading on Binance Testnet)

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. configure
cp .env.example .env
#   - BINANCE_TESTNET=true
#   - BINANCE_API_KEY=...      (from https://testnet.binance.vision/)
#   - BINANCE_API_SECRET=...
#   - ANTHROPIC_API_KEY=...
#   - TELEGRAM_BOT_TOKEN=...   (from @BotFather)
#   - TELEGRAM_ALLOWED_USER_IDS=123456789

# 3. run
python -m scripts.run_paper
```

The agent will:
- Subscribe to BTCUSDT, ETHUSDT, SOLUSDT kline + trade streams on Testnet.
- Run TWO strategies side-by-side:
  - **IndicatorConfluenceStrategy**: weighted-vote indicator scoring on 1m/5m/15m/1h. **No expected edge** on liquid majors after costs — this exists as a learning lab in paper mode. See `docs/ARCHITECTURE.md` for the honest analysis.
  - **FundingHarvestStrategy** (the real edge): polls Binance perp funding rates; when 8h funding crosses thresholds, opens a delta-neutral pair (long spot + short perp) sized to your `FUNDING_NOTIONAL_PER_PAIR_USD`. Closes when funding crosses back or basis blows up. Realistic 5–15% APY in calm markets, 20–40% during high-vol regimes.
- Position lifecycle is paper-simulated end-to-end: stops/TPs trigger on real bar data with realistic 25 bps adverse slippage on stops, 5 bps on TPs.
- Telegram sends signal cards for indicator trades and pair-open notifications for funding trades.
- LLM agents (advisory in current build) summarize news + propose strategy config tweaks for your review.

## Repository layout

```
src/
  agents/              LLM agents (Strategy, News, Anomaly, Telegram)
  config/              Settings, allowlist, strategy config schema
  models/              Pydantic models (Signal, Proposal, Position, etc.)
  services/            Storage, event bus, news, sentiment
  tools/               Binance client, indicators, signal generator,
                       risk gate, executor
  orchestrator.py      Main asyncio supervisor
scripts/
  run_paper.py         Paper trading entry point (testnet)
  run_live.py          Live trading entry point (mainnet, tiny size)
  backtest.py          Walk-forward backtest harness (stub)
tests/                 Pytest
docs/
  ARCHITECTURE.md      Full system design + failure modes
```

## Going live

After paper trades show edge over 100+ trades on testnet:

1. Generate **mainnet** API keys with `Enable Spot & Margin Trading` and
   `Enable Futures` (and **IP-whitelist** to your VPS).
2. Set `BINANCE_TESTNET=false` and `MAX_NOTIONAL_USD=50` (start tiny).
3. Re-run risk guardrails review in `src/config/settings.py`.
4. Deploy via `docker compose up -d` on your VPS.

## License

Personal use only. Not financial advice.
