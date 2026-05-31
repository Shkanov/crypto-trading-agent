---
name: project-levelbreak-validation
description: LevelBreakoutStrategy was built and FALSIFIED by validation on 2026-05-25 — flag stays off until someone re-validates against new params or a new universe.
metadata: 
  node_type: memory
  type: project
  originSessionId: 3f1b4043-15c2-428f-ba7f-e8c6aa160c54
---

`LevelBreakoutStrategy` (src/strategies/level_breakout.py) encodes a multi-TF
"break of prior-D1-level + HTF agreement + volume/momentum/vol filters"
pattern, inspired by the @aktradescalp Telegram channel. Trendline (наклонка)
variant included. Settings flag: `level_breakout_enabled` (default False).

**Validation run on 2026-05-25** via `scripts/levelbreak_validate.py`:

- **Cross-symbol, 15m×5000 bars perps**: all 7 symbols (BTC/ETH/SOL + FIDA/
  PROVE/BANANAS31/GRASS) lose money. Best case BANANAS31USDT at -5%
  annualized (≈noise). Alts show higher win rates (30-42%) than majors
  (27-35%) but cost drag still pulls them negative.
- **Walk-forward, 4×52d on BTCUSDT perps**: all 4 windows lose, win rates
  26-33%, no regime where the pattern works. Kills the "bad window" hypothesis.
- **Side-by-side on BTC 5000×15m**: indicator-confluence -$38 / mean-reversion
  $0 (no trades) / level-breakout -$54. Levelbreak takes 6× more trades than
  indicator-confluence for similar absolute loss.

**Why:** 2R target needs >33.3% win rate just for breakeven. The pattern hits
27-42% raw — fees/slippage push it under. The channel's apparent edge likely
comes from intra-day discretion the author has but doesn't publish (he skipped
half the calls on the sample day; says "перемудрил"). That's not encodable.

**How to apply:**
- Don't enable the flag without a *new* validation run (different params,
  different universe, or a code change to the rules).
- Don't parameter-tune to find a "working" backtest — it's overfitting
  against a small biased sample.
- The shipped code remains as a learning-lab module, same status as
  [[project_llm_supervisor_pattern]]'s indicator-confluence — honest negative
  result, not dead code.

Related: [[project_trader_agent_design]] (the broader LLM-as-trader paradigm
this would have slotted into).
