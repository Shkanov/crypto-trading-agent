---
name: project_cross_pair_execution
description: Native crypto-cross vs synthetic 2-leg execution comparison — 2026-05-31
metadata: 
  node_type: memory
  type: project
  originSessionId: a3cac948-1307-4702-89f1-8a72047feb89
---

User asked (2026-05-31) to explore strategies on crypto-CRYPTO pairs (not crypto-fiat). Chose "Both — compare": run the same relative-value signal two ways and compare net-of-cost edge. Built `scripts/backtest_cross_pairs.py`.

**Design:** identical β=1 ratio z-reversion signal (z-score of log(base/quote), entry|z|>=2, exit<=0.5, stop>=3.5) executed via (A) native cross spot e.g. ETHBTC = 1 instrument/2 fills per round-trip, vs (B) synthetic ETHUSDT/BTCUSDT 2-leg dollar-neutral = 4 fills, vs (C) synthetic on perps (5bp taker). Half-spreads calibrated live from bookTicker. Costs decomposed into raw_gross / spread / fee, with turnover-impact reported SEPARATELY.

**Live spread/liquidity snapshot (2026-05-31):** ETHBTC 1.83bp spread but only ~154 BTC/day (~$11M); SOLETH 1.22bp / ~$0.7M/day; vs ETHUSDT 0.02bp/$181M, BTCUSDT 0.00bp/$473M, SOLUSDT 0.6bp/$76M. Crypto-quoted books are TIGHT-quoted but LOW-turnover — liquidity migrated to USDT.

**Key cost-model gotcha:** the repo's Almgren impact `k·√(notional/ADV_5m)·1e4` explodes on low-turnover crosses (charged ~500bp/fill on a $1000 ETHBTC ticket) even though the quoted spread is 1.83bp. Low turnover ≠ thin top-of-book for an MM-quoted cross. Fix in the script: treat the cross at major impact tier (k=0.05) and keep turnover-impact as a separate sensitivity, not in headline net. **When backtesting any low-turnover-but-tight-spread instrument, don't trust turnover-based sqrt-impact — use the quoted spread.**

**Findings:**
- Execution-cost thesis CONFIRMED: native halves the fee count. Decisive for LOW-turnover strategies. SOL/ETH (40 round-trips/yr): native spot +$8 net vs synthetic spot −$109 vs synthetic perp −$29 — the $80 fee saving flips loss→breakeven.
- For HIGH-turnover (ETH/BTC, 194 RT/yr) fees dominate everything and the signal's negative raw gross (−$413) sinks all variants.
- The β=1 ratio-reversion SIGNAL is weak/negative on both pairs over trailing 365d (raw gross −$413 ETH/BTC, +$98 SOL/ETH). This was a plumbing comparison, not new alpha. The validated cointegration ([[strategy_tuning_sprint]]: pairs 1/2 ETH/BTC) is stronger because it β-fits + ADF-gates — which the native cross structurally CANNOT do (β is forced to 1).

**Verdict:** native crosses win on cost ONLY for small-size, low-frequency relative-value; they're capacity-capped (thin turnover), spot-only (no funding/perp harvest), can't optimize β, and need margin to short. For scalable stat-arb, synthetic PERP legs win (deep books, β-hedge, 5bp, funding).

**Full-cointegration follow-up (2026-05-31, committed):** added `--signal coint` to the script (rolling Engle-Granger β-fit + ADF gate). Ran on SOL/ETH:
- Validated harness (`backtest_pairs --pairs SOLUSDT:ETHUSDT`): −$1929, wr 6%, **coint 16/44 refits, ADF p̄=0.25** → SOL/ETH is NOT cointegrated. This is the pair that FAILED the sprint (memory said "pairs 1/2 ETH/BTC" — ETH/BTC passed, SOL/ETH failed; reconfirmed).
- Cross-pair script coint mode: cointegrated only 3696/8760 bars (~42%), **median β=1.04** (≈1, so β-freedom buys nothing here and native faithfully replicates the spread). raw gross ≈0 (native −$11, synth +$47) → no edge. Net: native spot −$704, synth spot −$1234, synth perp −$613.
- **Conclusion: the "stronger" cointegration signal does NOT rescue SOL/ETH** — no edge to begin with, and coint churns MORE (309 RT vs 40 in ratio mode) → bigger fee drag. Cost ranking holds: native spot ≈ synth perp < synth spot. Native's turnover-impact on SOLETH is $8k+ at 309 trades — uninvestable at size.
**ETH/BTC full cointegration (2026-05-31) — the decisive test (β≠1):**
- Validated harness: −$401, coint 14/44, ADF p̄=0.234 → even ETH/BTC is only WEAKLY cointegrated on this trailing 365d (cointegration regimes decay; it was the sprint keeper on 2026-05-27 data).
- Cross-pair coint: **median β=1.43** (materially ≠1) — so here the native β=1 mis-hedge BITES. Native raw gross +$136 vs synthetic +$216: native leaves ~37% of the spread edge on the table because β=1 isn't the cointegrating combination.
- Net: native spot −$91, synth spot −$247, **synth perp −$16 (≈breakeven, WINNER)**.
- **Final verdict: synthetic PERP legs are the right vehicle for cointegration stat-arb.** It gets the full β-hedged edge AND 5bp taker → higher raw edge survives fees. Native only beats synthetic SPOT (the worst vehicle) via fee savings, but (a) sacrifices edge when β≠1 and (b) can't access the cheap perp venue (no ETHBTC perp). Native crosses are justified ONLY for low-freq, small-size, β≈1 relative-value where perp isn't available.
- Caveat: all three slightly negative this year because ETH/BTC cointegration weakened (14/44); synth perp would likely turn positive in a more strongly-cointegrated window.

**Robustness sweep (2026-05-31) — BNB/ETH (β=0.47) + SOL/BTC (β=1.56):** confirms the ranking is UNANIMOUS across 4 pairs.
- BNB/ETH: native −$598 / synth-spot −$650 / **synth-perp −$274 (win)**. β=0.47, native raw +$81 vs synth +$107 (mis-hedge); BNBETH spread 2.8bp also hurts native.
- SOL/BTC: native −$216 / synth-spot −$447 / **synth-perp −$212 (win)**. β=1.56 → native raw goes NEGATIVE (−$5) vs synth +$36 — β=1 cross captures none of the edge.
- **Native mis-hedge severity scales with |β−1|:** SOL/ETH β1.04 (native fine) < ETH/BTC β1.43 (−37% edge) < BNB/ETH β0.47 (−24%) < SOL/BTC β1.56 (native edge negative).
- **ROBUST EXECUTION RANKING (all 4 pairs): synthetic perp ≥ native spot > synthetic spot.** Perp wins/ties every time. Synthetic spot always worst.
- Secondary finding: NONE of the 4 crypto-crypto pairs are profitable on trailing 365d (all cointegrated only 25-42% of bars) — the cointegration edge has broadly DECAYED vs the 2026-05-27 sprint window. Signal-timing/regime issue, separate from the (robust) execution-venue conclusion.

**Rolling-ADF timeline on ETH/BTC (2026-05-31, `scripts/_xpair_adf_timeline.py`, 60d window/daily step, 2024-07→2026-05):** cointegration is EPISODIC/regime-switching, NOT gradual decay. Overall 26% of windows cointegrated. Clear ON regime **2025-11 → 2026-03** (Nov 60%, Dec 48%, Jan 84%, Feb 100%, Mar 100%, ADF p→0.010, β stable ~1.2-1.5; longest streak 79d 2026-01-13→04-01). Then a SHARP BREAK at the start of **2026-04**: Apr 3% coint, May 0%, ADF p 0.19→0.33. β goes wild out-of-regime (range [-1.71, 4.25], even negative in 2025-09).
- This explains the losing trailing-365d backtest AND the sprint pass: the 2026-05-27 sprint backtested over a window dominated by the healthy Nov-Mar regime → passed; deploying now (late-May, in the broken regime since April) loses. Classic validated-in-regime / deployed-out-of-regime trap.
- The strategy's per-refit ADF gate (p<0.05) is too loose/slow: weekly refit lets positions enter late in a regime then stop out when it breaks mid-hold, and p<0.05 admits marginal/noisy windows. Also worth finding what broke ETH/BTC ~2026-04-01.

**Health gate prototype (2026-05-31, `backtest_cross_pairs.py` `--gate-p/--beta-window/--beta-tol/--persist-refits`):**
- Absolute strict ADF (p<0.01) is the WRONG lever — the repo's approximate `_adf_pvalue_via_ols` rarely dips below 0.01 on weekly 60d refits → 0 tradeable bars. β-stability alone barely moved ETH/BTC (β stayed ~1.4 through the April break; it was a stationarity break, not a β break).
- **PERSISTENCE is the lever that works**: require last N refits to all pass p<0.05 (trade only established regimes; stand aside at flickering onset; exit on first failing refit). ETH/BTC persist=4 (~4wk): synth-perp **−$16 → +$77/yr** (+7.7%), round trips 96→22 — gate isolates the Feb-Mar regime, avoids broken Apr-May.
- **But pair-selective — does NOT manufacture edge**: persist=4 leaves SOL/ETH −$592, BNB/ETH −$218, SOL/BTC −$221 (all still negative; their RT barely drop → they flicker through the gate without ever being tradeably mean-reverting). The gate only helps where a CLEAN regime existed (ETH/BTC).
- Honest caveat: persist=4 was picked from the in-sample sweep; ETH/BTC +$77 is only 22 trades (low N, fragile).

**CPCV+PBO verdict (2026-05-31) — FALSIFIED.** Added `persist_refits` to PairsParams + simulate_pair (backward-compat, default 1) and a `--persist-grid` sweep dim to cpcv_validate_pairs. Ran ETHUSDT:BTCUSDT, 2y (17520 bars), persist-grid 1,2,4 → 108 configs.
- **0/99 trading configs have positive IS-Sharpe.** The +$77 analog (lb1440/ze2.0/pr4) at the harness's SPOT costs: −$390 to −$492, IS-SR −1.7 to −2.3, OOS-SR −1.1 to −1.4. Best trading config by PnL −$111.
- The trailing-year +$77 was an artifact of THREE things stacking: PERP costs (5bp) not the harness's spot (10bp), one favorable 365d regime, and only 22 trades. Doesn't survive 2y CPCV. Even zeroing fees the signal IS-SR stays ≤0, so perp doesn't rescue it either.
- **The health gate does NOT make crypto-crypto cointegration deployable.** Classic validated-in-regime trap, now killed by CPCV. ETH/BTC cointegration genuinely decayed post-2026-04; no gate harvests an edge that isn't there.
- **Harness robustness bug found:** cpcv_validate_pairs PBO reported PASS (0.135) because its IS-best selector picked a 0-trade null config (Sharpe 0 > all-negative). PBO can be gamed by do-nothing configs.

**PBO null-config bug PATCHED (2026-05-31):** Two bugs, both fixed + tested.
1. *Null-config selection*: `src/services/cpcv.py pbo()` now DROPS non-trading (all-zero) columns before the argmax (active_mask param + auto all-zero detect; n_dead_columns on PBOResult; returns degenerate pbo=1.0/n_partitions=0 if <2 trading configs survive). Added `select_is_best_idx(sharpes, trades, min_trades)` so reporting also skips dead configs. Wired into ALL 5 cpcv_validate_* scripts (carry uses `weeks`). Input <2 cols still raises (contract preserved).
2. *Weak decision in pairs only*: `cpcv_validate_pairs` used PBO-only `_decision` while funding/meanrev already use the conjunction gate `_combined_decision` (PBO<0.5 AND IS-best OOS-mean>0). PBO alone passes uniformly-LOSING families (consistent badness = no selection bias). Aligned pairs to the combined gate.
- Regression tests added to tests/test_cpcv.py (21 pass): dead-column drop, select_is_best_idx skips non-trading, degenerate-all-dead → pbo=1.0.
- Re-ran ETH/BTC: now IS-best = a REAL config (lb2160/ze2.0/pr4, SR −1.008, 53 trades, OOS −0.515), 9 dead dropped, **DECISION: REJECT [OOS-mean −0.515 ≤ 0 (no edge)]** — the honest verdict. Confirms the health-gated crypto-crypto cointegration is dead, AND fixes the harness for all future validations.
