---
name: project_macd_cross_bot
description: "BitMEX MACD-cross bot analyzed + backtested 2026-05-31 — FALSIFIED, dead on costs"
metadata: 
  node_type: memory
  type: project
  originSessionId: a3cac948-1307-4702-89f1-8a72047feb89
---

User dropped `BitMex trading bot.rar` (Krotov's "BitMEX simple trading robot", Habr/Medium 2018) for analysis + backtest on 2026-05-31.

**The bot's entire alpha** (strategy.py, identical across all 4 bundled variants): MACD(fast=8, slow=28, signal=9) histogram zero-cross on 1h XBTUSD. `hist[-2]<0 & hist[-1]>0` → long; `hist[-2]>0 & hist[-1]<0` → short. Single instrument, no stop, no edge filter. AWS "working" variant adds ±$100 fixed TP and flip-on-opposite-signal; Telegram alerts; 3 BitMEX accounts (master/long/short).

**Two defects found in the shipped code:**
- Original master variant computes MACD on `close.values` while fetching `reverse=True` (newest-first) WITHOUT reversing → MACD ran on time-reversed series, `hist[-1]` = OLDEST bar. Effectively traded noise. "new"/AWS variants fixed with `[::-1]`.
- `working solution from AWS/trader/main_loop.py` contains **live BitMEX API key/secret (3 accounts) + a Telegram bot token in plaintext**. Flagged to user to revoke.

**Backtest verdict: FALSIFIED.** Built `scripts/backtest_macd_cross.py` (faithful always-in-market reversal, costs ON = perp taker 5bps + half-spread + Almgren-Chriss impact on both legs of every flip; reuses repo cost/stats helpers; causal closed-bar MACD). Stop rule mirrors Card 1 Δfunding.
- Headline BTCUSDT 1h ~365d: 773 flips, net -$1420 (-142%/yr), even ex-fees -$647, Sharpe -3.51, DSR 0. DEAD.
- Robustness grid BTC/ETH/SOL × 1h/4h: **0/6 positive net of costs**. Best (ETH 4h) +$176 ex-fees but -$212 after $388 fees. Win rate ~30% everywhere.
- Root cause: fixed-param MACD zero-cross is the most over-published TA signal; whipsaws in range regimes (~770 flips/yr on 1h), and fees alone (~$770/yr at 1x on $1k) exceed any trend capture. Did NOT proceed to CPCV grid (correctly stopped per rule).

Consistent with prior single-asset indicator falsifications ([[project_levelbreak_validation]], funding 0/38). See [[strategy_tuning_sprint]] for what DID survive (carry, pairs).
