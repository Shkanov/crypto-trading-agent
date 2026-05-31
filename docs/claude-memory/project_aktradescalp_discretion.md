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

**OI-persistence reversal test (2026-05-31) — FALSIFIED.** Channel msg 174 longed ESPORTSUSDT after a dump citing "ОИ не упал, объемы тоже" (OI + volume held = no capitulation → revert). Built `scripts/backtest_oi_persistence.py` (Binance futures_open_interest_hist, only ~30d available → smoke). Event study over 20 perps: dumps where OI HELD did NOT revert — +12h mean −0.57% to −3.25%, win 14-45%; the costed long-the-dump sim lost −$626 to −$2560 (win 7-12%). If anything the OPPOSITE leans true (capitulation/OI-fell dumps reverted more at +24h, but n=5, noise). Economic read: price drop with OI holding = new shorts piling in = continuation, not reversion. The single ESPORTSUSDT win was an anecdote, not a mechanical edge. Caveat: 30d data cap, small N — directional not conclusive; a real test needs OI captured live forward. Consistent with the core finding: his edge is discretionary selection, not any single mechanizable rule.
