---
name: project-levelbreak-continuation
description: "Resting state of the @aktradescalp / LevelBreakoutStrategy thread — where the work paused 2026-05-25, what's uncommitted, what's ephemeral, what comes next."
metadata: 
  node_type: memory
  type: project
  originSessionId: 3f1b4043-15c2-428f-ba7f-e8c6aa160c54
---

Session checkpoint 2026-05-25 (paused before a planned laptop restart).
Combine with [[project-levelbreak-validation]] and
[[project-aktradescalp-discretion]] for the full picture.

## Resting state

The conversation arc: user asked to encode the @aktradescalp Telegram
channel's pattern. I built `LevelBreakoutStrategy` (D1-break + HTF + filters,
incl. trendline variant), validated it (loses across 7 symbols × 4 walk-
forward windows on default params), then investigated the channel author's
discretion (109 parsed messages, 36 replayed calls — found his alpha is
~+180 bps/trade above random entries, lives in symbol+timing selection, not
in the rules we encoded). Conversation paused after that summary.

No in-flight work. All 14 tasks complete. Strategy flag stays OFF.

## Code state — committed before restart 2026-05-25

All strategy + validation work landed in commit `f8c64b0`:
"strategies: add LevelBreakoutStrategy + validation harness"

Files in that commit:
- `src/strategies/level_breakout.py`, `tests/test_level_breakout.py` (new)
- `scripts/_probe_symbols.py`, `scripts/levelbreak_validate.py` (new)
- `scripts/backtest.py`, `src/config/settings.py`, `src/orchestrator.py`,
  `src/services/backtest.py` (modified — `levelbreak` CLI choice, flag,
  orchestrator wiring, `backtest_level_breakout` + `simulate_level_breakout`)

Tests: 41/41 (12 new). Flag `level_breakout_enabled` defaults False.

## Research artifacts — preserved under `data/research/aktradescalp/`

(Gitignored via `data/` rule. Persists across restart on disk.)

- `pages/page_*.html` (8 pages of raw channel HTML from t.me/s/aktradescalp)
- `aktradescalp_messages.json` (109 parsed messages)
- `aktradescalp_calls.json` (36 dated entry-calls extracted for replay)
- `parse_aktradescalp.py`, `analyze_aktradescalp.py`, `replay_calls.py`,
  `fetch_aktradescalp.sh`

To re-run the full pipeline:
```
cd data/research/aktradescalp
bash fetch_aktradescalp.sh        # ~30s, refetches latest HTML
python parse_aktradescalp.py      # outputs aktradescalp_messages.json
python analyze_aktradescalp.py    # corpus stats + writes calls.json
python ../../../replay_calls.py   # path-adjusted: replay vs random baseline
```
(`replay_calls.py` reads from `/tmp/aktradescalp_calls.json` by default —
will need a path edit if running from the new home.)

## Next directions discussed but NOT implemented

Both are bigger than this PR's scope; user signaled neither yet.

1. **Market-scanner agent** — closer fit to the author's actual discretion
   than more rule-encoding. Scans low-cap perp tape during 07-12 UTC for
   unusual vol_z + clean breakout structure, surfaces candidates. Fits the
   trader-agent paradigm from [[project_trader_agent_design]].

2. **ScalpSignalAgent (channel-as-narrative-input)** — watch t.me/s/
   aktradescalp via the parsed-page approach (or a proper MTProto client,
   per the earlier conversation), surface new messages on Telegram for
   manual review. Never auto-fire. Treats him like NewsAgent does CryptoPanic.
   Has follower-latency tax; only worth it if his messages are still mostly
   actionable at +60s.

## What NOT to do (reminders)

- Don't enable `level_breakout_enabled` — pattern was falsified on default
  params AND the discretion replay showed the rule captures ~5% of his calls.
- Don't tune the strategy params against the 36-call corpus — overfitting
  against a small biased sample.
- Don't pipe the channel into the system as an auto-firing signal — see
  [[project-aktradescalp-discretion]] caveats (n=26 replay, EDEN missing,
  best-case fills).
