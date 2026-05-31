---
name: project_carry_robustness
description: "Funding-carry stress test 2026-05-31 — validated PASS is selection-luck, not robust signal"
metadata: 
  node_type: memory
  type: project
  originSessionId: a3cac948-1307-4702-89f1-8a72047feb89
---

Stress-tested the one deployable futures strategy (funding carry) on 2026-05-31 after finding Pairs cointegration had silently decayed. Carry is the [[strategy_tuning_sprint]] "PASS" (reported +$204 / Sharpe 0.81 on 05-27 PIT dataset).

**Health-check (`scripts/_carry_health_check.py`):** pinned-05-27 universe now 365d −$120 / 90d −$171 / 30d +$56; live top-30 now 365d +$71 / 90d −$31 / 30d +$20. Sign FLIPS by universe; the trailing-90d (≈Mar–May 2026) is negative for both — the same window Pairs broke, so the whole relative-value book decayed spring 2026. Last 30d positive (regime recovery, tiny sample).

**Universe-robustness sweep (`scripts/_carry_universe_robustness.py`, 60 random 30-coin universes from top-60, 365d, costs ON, validated top3 params):** mean PnL −$4.79, median −$25.36, **std $145.64**, range −$274..+$448, **only 43% positive**, Sharpe mean −0.09. The pinned-05-27 "validated" universe = **18th percentile** (the +$204 was a lucky high-percentile draw; re-drawn today it's below median); live top-30 = 70th pct. **VERDICT: SELECTION-DRIVEN** — carry PnL straddles zero ~50/50 with spread ≫ mean, so the "edge" is mostly which coins you picked (hindsight `build_universe` = today's volume leaders applied backward), NOT a reliable signal. The sprint PASS was one lucky draw.

**Implication:** do NOT size up carry as-is (top-3). Candidate fixes (reduce selection variance): (a) more legs per side; (b) TRUE point-in-time universe instead of today's leaders backward.

**More-legs fix WORKS (2026-05-31, same 60 draws/seed):**
| legs/side | mean PnL | median | std | %positive | Sharpe>0.5 | Sharpe<−0.5 |
| top-3 | −$5 | −$25 | $146 | 43% | 20% | 30% |
| top-5 | +$109 | +$79 | $171 | 68% | 50% | 12% |
| top-8 | +$64 | +$73 | $108 | 68% | 57% | 17% |
Diversifying from 3→5/8 legs flips the distribution from coin-flip-around-zero to **68% positive, median clearly >0, far fewer disaster draws** (top-8 bad-Sharpe 30%→17%, lowest std $108). top-8 is the most STABLE (lowest spread, 57% sizeable Sharpe); top-5 highest mean but wider. NOT bulletproof — ~32% of universes still lose, p10 negative — so it's a real but WEAK edge: deployable at modest size with 5–8 legs, not high-conviction. Verdict heuristic updated to "LEANING POSITIVE" for this 62–80%-positive band.

**PIT-universe sweep (`scripts/_carry_pit_universe_sweep.py`, top-8 legs, as-of-time volume ranking within random 50-coin sub-pools, 365d) — the more-legs win was MOSTLY HINDSIGHT:**
| sweep | %positive | median | Sharpe mean |
| hindsight top-3 | 43% | −$25 | −0.09 |
| hindsight top-8 | 68% | +$73 | +0.37 |
| **PIT top-8** | **38%** | **−$28** | **−0.31** |
Once volume is ranked as-of-time (no hindsight), carry falls back to ~38% positive / negative median — the 68% from the hindsight more-legs sweep came largely from using TODAY's volume leaders. The deterministic full-pool PIT (+$109/0.80, the earlier "PIT survives" re-validation) is the **93rd percentile** of the sub-pool distribution = a lucky outlier, not typical. Still optimistic (pool = top-80 by CURRENT vol = residual survivorship; true delisted correction → worse).

**Allocator carry-floor audit + FIX (2026-05-31):** the allocator had NO explicit carry floor, but its evidence floor (`min_active_days=30`) screened on ACTIVITY not edge → it zeroed the event-driven sleeves (pairs 14d, meanrev 13-20d active/yr) and handed **carry 100%** under all 3 methods (equal/inv_vol/hrp identical SR 2.06) — a full bet on the stale lucky-draw carry. Fixed in `portfolio.py`: added `_apply_edge_gate` (zero sleeves with realized window Sharpe < `min_sharpe`, blank to cash if none) + `min_surviving_sleeves` cash guard + `AllocationResult.deployable`. Wired into settings (`allocator_min_sharpe=0.0`, `allocator_min_surviving_sleeves=1`) + orchestrator. Backward-compat (defaults -inf/1 are no-ops); 35 portfolio tests pass incl. 4 new. Re-run (`backtest_allocator --min-sharpe 0`): OLD=carry 100%; edge-gate=follows realized edge per-window (no more auto-carry); edge-gate + min-surviving=2 → **cash 11/11 rebalances** (no 2 sleeves ever have concurrent positive edge → honest "nothing diversified to deploy"). Cash handled safely downstream (share=equity·w → 0).

**FINAL VERDICT — carry is NOT a robust standalone edge on this universe.** Validated PASS = selection luck; more-legs helped only under hindsight; honest PIT leaves it weak-to-negative. **Do NOT ship carry-heavy in the allocator.** At best a tiny diversifier with the full broad pool + many legs, sized for a weak/uncertain edge. Mirrors the Pairs decay ([[project_cross_pair_execution]]): the whole relative-value book is shaky on current data. Three `_`-prefixed diagnostics: _carry_health_check, _carry_universe_robustness (--top-n), _carry_pit_universe_sweep.
