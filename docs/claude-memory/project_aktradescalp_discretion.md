---
name: project-aktradescalp-discretion
description: "Investigation of what @aktradescalp's discretion adds on top of his stated rules — concludes the alpha is in symbol+timing selection, not in the mechanical setup."
metadata: 
  node_type: memory
  type: project
  originSessionId: 3f1b4043-15c2-428f-ba7f-e8c6aa160c54
---

Followup to [[project-levelbreak-validation]]. Pulled 109 messages from
t.me/s/aktradescalp (Mar 30 – May 22, 2026), classified them, and replayed
his 36 dated entry-calls on real perps M5 klines with our backtest's
stop/TP/cost model.

**Channel profile (the parts that aren't in the published rules):**

- **Active only 35% of calendar days** (19 of 54). Mean 2.3 calls per active
  day, median 2, max 5. He's a scanner, not a daily-grind trader.
- **Short-biased: 57% short / 41% long.** Contradicts the channel tagline
  "только колы и отработка".
- **Time-of-day concentration: 10–15h Moscow (07–12 UTC).** 11h MSK alone is
  25% of all entries. Effective ~5h trading window.
- **Friday-heavy: 14 of 44 entries on Fridays.**
- **TF mix is broader than encoded:** 64% of entries have NO TF tag, 20%
  M15-only, 9% M5-only — only ~5% are the D1+M5 break our strategy targets.
- **Symbol universe: 28 unique tickers across 52 mentions in 44 entries.**
  Almost a new symbol per call. ETHUSDT appears 2×, BTC never. He trades
  low-cap alts (PLAY, AGT, NAORIS, ZKJ, BLESS, PRL, PIPPIN, LAB...).
- **Setup mix: 61% breakout, 14% trendline, 7% combo, 18% bare directional
  call** with no published setup at all.

**Replay backtest result (2026-05-25, n=26 valid of 36 — 10 EDEN/KAT/PRL/BILL
calls skipped due to testnet symbol unavailability):**

- His calls: **65.4% WR, +164.8 bps/trade mean, +4285 bps total**.
- Random baseline (same symbol+side, random t in 07-12 UTC): **46.2% WR,
  -13.7 bps/trade mean, -356 bps total**.
- Delta: ~+180 bps/trade — large enough to indicate real discretion alpha.
- Holds on both sides: long delta ≈ +163 bps/trade, short delta ≈ +193.

**Why:** his apparent edge is **symbol-selection + timing**, not the setup
rule we mechanized. Random entries on the same alts in the same hour lose
money; his entries make money. The mechanical D1-break rule captures ~5%
of his actual calls; the rest is M15/freeform on whatever ticker is moving.

**How to apply:**
- Confirms [[project-levelbreak-validation]]: do NOT enable
  `level_breakout_enabled`. Pure mechanical rule has no edge on this
  universe; the channel's alpha lives in selection we can't replicate from
  static rules.
- If we ever try to capture some of this: build a market-scanner agent
  (low-cap alts with unusual vol_z/movement in 07-12 UTC), not more
  signal-rule encoding. Or treat the channel as a narrative input (like
  NewsAgent), not a trading filter.
- Caveat: small N (26), wide CI on 65% WR (~45-83%); EDEN (his most-mentioned
  symbol) had no replay data; his real PnL with tighter discretionary
  scratches may differ from our 1×ATR / 2R / 24h-stop replay.

Artifacts kept under /tmp during this session:
- /tmp/aktradescalp_messages.json (109 parsed messages)
- /tmp/aktradescalp_calls.json (36 entry-calls)
- /tmp/replay_calls.py, /tmp/analyze_aktradescalp.py, /tmp/parse_aktradescalp.py
(Note: these now live committed under data/research/aktradescalp/, refreshed through 2026-05-31 = 128 msgs.)

**REFRESH 2026-06-07 (corpus → 142 msgs / 63 entries, through Jun 5; replay n=38).**
Profile unchanged on bigger n: 52% short, **65% no-TF**, breakout 59% / trendline
19%, 40 symbols / 77 mentions, active 44% of days — D1-break still ~5% of calls.
Fresh replay (same 1×ATR/2R/24h model): **his calls 52.6% WR, +93.5 bps/trade,
+3552 total; random same-sym+side+window 23.7% WR, −68.9 bps/trade.
Delta ≈ +162 bps/trade** — the discretionary TIMING edge PERSISTS (random holds
symbol+side fixed and still loses, so it's entry timing, not just symbol pick).
16 low-caps skipped (testnet missing EDEN/BTW/ZEST/USELESS/etc. — likely his best).

**OOS REPLAY 2026-06-07 — SELECTION EDGE INVERTS OUT-OF-SAMPLE (do not build).**
`data/research/aktradescalp/oos_replay.py` (mainnet 5m, fixes testnet symbol
gaps, same 1×ATR/2R/24h model as replay_calls, split at 2026-05-27 = first day
after the W3 tuning window):
- IN-SAMPLE (<5-27, n=38): his 53% WR +73 bps vs random 21% −89 bps → **+162 bps**.
- OOS (>=5-27, n=16): his **12% WR −159 bps** vs random 38% **+72 bps** → **−231 bps**.
His calls LOST OOS and random beat them. Third fragility signal (joint-sim W3
fail → full OOS inversion). Caveats: small OOS n=16 (t−1.65), and his recent msgs
are "watching" alerts not immediate entries, so timestamp-immediate replay
mis-times him more now — but the SAME model flipped +162→−231, and the
automatable joint-sim already failed its last window, so mechanically copying the
channel is NOT a stable edge. **Decision: do NOT build the microstructure-entry
gate** (would overfit an OOS-negative thread); validating first prevented that.
Channel remains a narrative/idea input only, never an auto-fire signal.

**Entry method (May-Jun refinement — what he's doing, NOT a deployable rule):** he now (1) waits for
**наторговка** = consolidation AT the level before a breakout ("жду наторговки
хаёв или слома"), not the first touch; (2) triggers on **закол дневки** = a
liquidity *sweep* of the daily high/low then go; (3) gates on **OI+volume as
fuel** ("ОИ не упал, объёмы — тоже" → reversal idea; "объём и ОИ падает → зона
опасная, наблюдать"; "кто-то откупает → стопы есть"); (4) more 1m/3m scalps.
These three filters ARE the "microstructure entry" flagged as the #1 improvement
to [[project_cascade_strategy_research]] (Sharpe 2.3-2.6/80d but W3 fail). Next
step under the stable-max-PnL directive: encode consolidation-at-level + daily
sweep + OI/vol-holding as the **cascade strategy's entry gate**, validate via
CPCV — NOT re-encode the falsified mechanical breakout rule. Do not auto-fire the
channel (follower-latency tax; his best symbols are illiquid).

**OI-persistence reversal test (2026-05-31) — FALSIFIED.** Channel msg 174 longed ESPORTSUSDT after a dump citing "ОИ не упал, объемы тоже" (OI + volume held = no capitulation → revert). Built `scripts/backtest_oi_persistence.py` (Binance futures_open_interest_hist, only ~30d available → smoke). Event study over 20 perps: dumps where OI HELD did NOT revert — +12h mean −0.57% to −3.25%, win 14-45%; the costed long-the-dump sim lost −$626 to −$2560 (win 7-12%). If anything the OPPOSITE leans true (capitulation/OI-fell dumps reverted more at +24h, but n=5, noise). Economic read: price drop with OI holding = new shorts piling in = continuation, not reversion. The single ESPORTSUSDT win was an anecdote, not a mechanical edge. Caveat: 30d data cap, small N — directional not conclusive; a real test needs OI captured live forward. Consistent with the core finding: his edge is discretionary selection, not any single mechanizable rule.
