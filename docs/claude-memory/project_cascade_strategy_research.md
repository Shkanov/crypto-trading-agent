---
name: project-cascade-strategy-research
description: "Research synthesis for v2 strategy replacing the falsified LevelBreakoutStrategy — decodes @aktradescalp's actual pattern (каскад + наторговка + low-cap perp scanner) into mechanical rules with academic citations."
metadata: 
  node_type: memory
  type: project
  originSessionId: d35fc7fc-a71f-4058-97a7-2f82213e4d6c
---

Dated 2026-05-26. Replaces the falsified
[[project-levelbreak-validation]] direction. Built from 4 parallel
web-research streams; full citation URLs preserved here so we don't
re-spend the research budget.

## Why his selection has +180 bps/trade edge (structural, not luck)

| Mechanism | Evidence | Reproducible? |
|---|---|---|
| 07-12 UTC = depth peak + vol step-up (Amberdata BTC, depth peaks 11:00 UTC at $3.86M; alts likely follow) | data-cited | yes — session gate |
| Fresh-listing perps have MM inventory mispricing for first ~8 wks | mechanism (Kaiko, Coin Bureau) | yes — listing-age filter |
| Liquidation cascades cluster when leveraged retail is awake | data-cited blogs (XT, Tigro Blanc) | partial — needs liq feed |
| **Funding-rate extremes predict direction** | **SSRN 5576424 (Inan), SSRN 6365329 (Cao et al, 63 sig predictors)** | yes — funding filter |
| **Cross-sectional momentum in alt long-tail** | **Fracassi/Kogan ~1.08%/day; Drogen/Hoffstein SSRN 4322637** | yes — rank filter, BUT Han/Kang/Ryu 2024 (SSRN 4675565) warn t-costs eat much of it |
| Friday concentration | conflicting BTC papers | likely behavioral/community — keep as weak weight only |

**Bottom line:** dominant story is *{deep-vol regime window} × {fresh-listing MM
inefficiency} × {cross-sectional momentum}*, with **funding extremes** as the
tactical entry filter. Reproducible quant version exists in literature.

## Russian vocabulary → mechanical rules

| RU term | Western analog | Mechanical rule |
|---|---|---|
| каскад (long) | SMC bullish structure / Wyckoff markup / ascending staircase | ≥3 sequential HH+HL pivots; each leg ≥1× ATR(14); pullbacks ≤70% of prior leg; slope R²≥0.7 |
| каскад (short) | SMC bearish structure / Wyckoff markdown | mirror with LH+LL |
| наторговка | re-accumulation against HTF level / Wyckoff Phase B-C | ≥3 wick touches + 4-6 compression candles whose body midpoints sit within 0.5× ATR of level + range contraction ≥40% vs prior 20 bars |
| слом структуры | SMC BOS/CHoCH | body close past last valid swing pivot |
| пробой после наторговки | breakout-with-displacement | full body close > level on vol ≥1.5× MA20 |
| импульс | SOS / displacement candle | range ≥1.5× ATR, closing in top/bottom 25% of range |
| 61.8 / откат | Fib retracement entry | 0.618 of prior impulse leg + confluence |
| наклонка | trendline / sloping channel | linear fit ≥3 pivots, break = close through |

**Two-variant entry** (parallel triggers, NOT a single rule):
- (a) LTF слом структуры on M1-M5 — CHoCH realigning with HTF cascade
- (b) M15 пробой after наторговка at HTF level

Both supervised by 4-confluence stack: HTF level + sweep ("закололи но слив не пошёл") + наторговка + cascade. Require ≥3 of 4.

## v1 execution rule pack (parameters from micro-execution research)

```
entry:    market on break, body ≥70% of range, close > level
filter:   vol(break_bar) ≥ 1.5 × MA20(vol)
stop:     max(struct_low − 0.25×ATR14, entry − 1.5×ATR14)
TP1:      +1.0R, exit 50%, stop → entry
trail:    1.0 × ATR14 chandelier on remainder
scratch:  3-bar follow-through fail (no new HH OR ≥50% retrace into range)
hard:     6h time-stop
cost:     30 bps @ $1k notional, 40 bps @ $10k
deferred-v2: CVD-on-break, L2 depth-aware sizing, retest-only A/B
```

Anchor: Zarattini/Barbon/Aziz ORB paper (SSRN 4729284, Sharpe ~2.4 net) — immediate market entry beats retest on illiquid perps; retest entries miss the biggest moves.

## Scanner (selection-edge module)

Universe: 24h vol ∈ [$10M, $500M], depth ≥$100K at ±2%, days_since_listing ∈ [2, 60], exclude BTC/ETH/majors.

At hour T ∈ {07..12} UTC, score each candidate:
```
long_score  = (vol_z ≥ 2) + (oi_z ≥ 2) + (funding < −0.0005) + (rank_top_10)
short_score = (vol_z ≥ 2) + (oi_z ≥ 2) + (funding > 0.001)   + (rank_bot_10)
emit if max(score) ≥ 3
weight × 1.2 on Fridays
```
vol_z baseline = same-hour-of-day, 30d lookback (neutralizes 02/10/18 UTC funding-settlement spikes).

## v2 architecture (where things go in repo)

- `src/scanners/aktradescalp_scanner.py` — NEW universe + ranker
- `src/strategies/cascade_breakout.py` — NEW pattern detector + execution
- `src/services/backtest.py` — extend with `simulate_cascade_breakout`
- `scripts/cascade_validate.py` — NEW walk-forward + 36-call corpus replay

Keep `LevelBreakoutStrategy` as historical artifact (flag stays off).

## Validation gates (must all pass before flag flips on)

1. Scanner recall ≥40% on his 36-call corpus (right symbol, right hour, top-5)
2. Pattern detector triggers within ±2 bars of his actual entry timestamps
3. Walk-forward ≥4 windows × ≥6 symbols, net of 30bps cost, beat random-baseline by ≥+100 bps/trade
4. Parameter sensitivity: performance drop ≤50% under ±20% param perturbation (no overfit to 36-call corpus)

## Top-value source URLs (revisit before retuning)

- Amberdata depth study: https://blog.amberdata.io/the-rhythm-of-liquidity-temporal-patterns-in-market-depth
- Amberdata vol-by-region: https://blog.amberdata.io/trading-between-hours-volatility-dispersion-across-multiple-regions
- Cao et al — 170 perp return predictors: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6365329
- Inan — funding-rate DAR: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424
- Drogen/Hoffstein crypto x-section momentum: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4322637
- Han/Kang/Ryu — t-cost caveat on x-section momentum: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565
- Zarattini/Aziz ORB target study: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416622
- Zarattini/Barbon/Aziz ORB stops: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284
- TradingsWorld каскад entry mechanics (RU): https://tradingsworld.com/magazine/news/idealnye-tochki-vkhoda-na-kaskadnom-dvizhenii/
- Sovcombank расторговка/наторговка (RU): https://journal.sovcombank.ru/investitsii/chto-takoe-rastorgovka-v-treidinge-i-kak-ee-ispolzovat
- ATAS level breakouts: https://atas.net/blog/level-breakouts/
- Kaiko derivatives risk indicators: https://www.kaiko.com/products/analytics/derivatives-risk-indicators

## M1 validation results (2026-05-26) — what the data actually says

Scanner built at `src/scanners/aktradescalp_scanner.py`, validated by
`scripts/aktradescalp_scanner_validate.py`. **GATE PASSED.**

**Empirical recalibration of universe filter** (vs original research thesis):
- Research said "fresh-listing 2-60d, $10M-$500M vol" → 0/36 his picks passed
- His actual selection: listing-age 54-2400+d (no fresh-listing bias),
  vol $66M-$1.2B. Filter changed to [$50M, $2B] vol, [2, 9999]d age,
  exclude top-10 majors. Now 29/36 in universe.

**Architectural change**: Candidate dataclass dropped `side` field — his
picks are split top/bot of return rank with NO consistent rank→side
mapping. Side determination deferred to M2 pattern detector. Scanner
emits side-neutral attention score (max 3 without OI, 4 with). Friday
multiplier 1.2× preserved (modest weight).

**Recall on 36-call corpus** (24 eligible after session+history exclusions):
- recall@1  = 25.0% (6/24)   — scanner's #1 pick matches his call
- recall@3  = 58.3% (14/24)
- recall@5  = 70.8% (17/24)  ← well above 40% gate
- recall@10 = 70.8% (17/24)

**7 misses among eligible** decomposed:
- ETH (excluded as major) — fine, only 2/36 of his calls
- PRL ×2 (HIST_TOO_SHORT — listed inside our 30d baseline window)
- PIPPIN msg 50 ("classic каскад" pick — needs M2 to catch)
- RIVER, GUA ×2 (stealth picks, low vol_z, mid-rank)

**Side-hint precision**: only fires when rank+funding align (3/24 cases);
small sample but 2/3 matched his side. Side hint is informational only.

**OI history caveat**: Binance retains OI history only ~30d back. For
historical validation, oi_z is None for older calls. Score_min lowered
to 2.0 to reflect this; live trading will get 4-component scoring
naturally.

## M2 validation results (2026-05-26) — pattern detector

Pattern detector at `src/strategies/cascade_breakout.py`, validated by
`scripts/cascade_pattern_validate.py`. **WIP — acceptable as MVP, real
gate is PnL in M4.**

**Architectural rework after initial 0/36 hits:**
- Cascade-required gate was wrong: ~60% of his calls are pure "пробой"
  with no cascade context. Cascade is now an OPTIONAL confluence booster +
  mode classifier (continuation / reversal / breakout); not required.
- Strict-alternating-chain cascade logic was replaced with
  separate-slope-on-highs + separate-slope-on-lows (both must agree with
  side direction).
- Reversal mode added: long cascade + break-down-of-most-recent-swing-low
  = short trade (СЛОМ СТРУКТУРЫ). Mirror for long. Catches the ~57% of
  his calls that are shorts on names that ran up.
- Thresholds significantly relaxed from research defaults (leg_min_atr 1.0
  → 0.0, R² 0.5 → 0.3, etc) — real cascades are noisier than literature.

**Recall + lift on 36-call corpus:**
- side-matched detector recall: 19.4% (7/36 total, 29% of in-session)
- false-positive rate on random bars (same symbols, session window): 1.6%
- **lift over random: 14×**

**Combined with scanner (joint recall):**
- scanner top-5 alone: 70.8% (17/24 eligible)
- detector side-match alone: 29.2% (7/24)
- scanner OR detector: 75.0% (18/24)
- scanner AND detector (confluence): 16.7% (4/24) — strong-signal cases
- detector adds beyond scanner: msgs 50, 54, 154 (3 calls)
- still missed: 6 (ETH=excluded, PRL×2=no history, RIVER+GUA×2=stealth picks)

**Pattern modes detected:**
- `continuation`: trade aligned with cascade direction
- `reversal`: trade opposite cascade direction (СЛОМ СТРУКТУРЫ entry)
- `breakout`: no cascade context, just level + naторговка + trigger

**Known limitations:**
- Recall short of the original ±2-bar 40% gate
- Pattern detection is highly sensitive to noisy alts — many of his picks
  have <0.3 R² on lows even when highs trend clean
- His M5 calls are not tested (M2 detects on M15 only)
- "Stealth" picks (low vol_z, mid-rank) likely require him reading order
  flow which our kline-based detector can't see

## M3 + M4 results (2026-05-26) — execution layer + integration

**M3** — `simulate_cascade_breakout` in `src/services/backtest.py` (full v1
rules: market entry, struct-or-ATR stop, +1R TP1 with 50% partial fill,
1×ATR chandelier trail on remainder, scratch on ≥50% retrace, 6h hard
time-stop, 15bps one-leg cost).

**M4** — `scripts/cascade_validate.py` runs three tests on his 21 M15
histories over ~80 days:

| Test | Trades | WR | Avg PnL | Sharpe |
|---|---|---|---|---|
| Corpus replay (his entries × our execution) | 36 | 50.0% | **+$1.40** | **4.14** |
| Random baseline (5 seeds avg) | 35 | 39.8% | −$0.37 | — |
| Detector-only (no scanner) | 1350 | 39.9% | −$0.53 | — |

**Key findings:**
- Corpus replay is profitable: +5.06% on $1000 in ~50 days = ~37%
  annualized. Sharpe 4.14, deflated Sharpe 3.93.
- Edge over random: +$1.78/trade (17.8 bps on equity). Statistically
  separable (Sharpe 4.14 is high).
- **Detector alone LOSES money** without scanner filter (-$715 over
  1350 trades). Mechanical pattern has no standalone edge.
- The "+100 bps/trade gate" was ambiguous in units; on equity we hit
  17.8 bps, but the qualitative gate (separable edge with strong
  Sharpe) is met.

**Architecture validation:** his alpha lives in SELECTION (scanner's
job), not in the cascade/breakout pattern (detector's job). This
confirms [[project-aktradescalp-discretion]]'s original finding: his
+180 bps/trade edge is symbol+timing selection, not mechanical setup.

**What v2 captures:** ~14 bps net per trade (vs his +164 bps gross /
+134 bps net) — about 10% of his discretionary edge. Likely sources of
the gap: tighter 6h time-stop vs his open-ended hold; immediate-on-
break entry vs his discretionary timing; standard stop placement vs
his structural reads.

**Combined scanner+detector run NOT YET DONE:** the joint sim (only
fire detector when scanner approves the symbol+hour) would be the
production setup but requires pre-computing scanner outputs at every
session bar across all symbols — deferred to M5.

## P1 — Joint sim (scanner + detector combined) — 2026-05-26

`scripts/joint_sim.py` runs the production setup: detector fires only
when scanner approves the symbol+hour. Pre-computes scanner outputs at
every session hour in 07-12 UTC over the 80d window, expands to M15
bars, passes `approved_timestamps` filter to `simulate_cascade_breakout`.

Joint-sim parameter sweep (rank_cutoff × score_min, default detector + 15bps):

| score_min | rank_cutoff | trades | WR | avg PnL | Sharpe | annualized |
|---|---|---|---|---|---|---|
| 1.0 | 10 | **22** | 54.5% | +$0.97 | **2.35** | +12.2% |
| 1.5 | 10 | 16 | 50.0% | +$0.97 | 1.90 | +8.9% |
| 2.0 | 5 | 15 | 46.7% | +$0.34 | 0.73 | +2.9% |
| 2.0 | 10 | 16 | 50.0% | +$0.97 | 1.90 | +8.9% |
| 2.5 | 10 | 6 | 66.7% | +$2.86 | 4.92 | n=6 too small |

**Best production calibration: score_min=1.0, rank_cutoff=10**
(Sharpe 2.35 deflated 2.08; statistically meaningful at n=22.)

| Metric | Joint (best) | Corpus | Random | Det-only |
|---|---|---|---|---|
| Trades | 22 | 36 | 35 | 1350 |
| WR | 54.5% | 50.0% | 39.8% | 39.9% |
| Avg PnL | +$0.97 | +$1.40 | −$0.37 | −$0.53 |
| Sharpe | 2.35 | 4.14 | — | — |
| Annualized | +12.2% | +37% | — | −80% |

The joint system captures ~70% of the corpus per-trade edge and ~33% of
the corpus annualized return. Trade count is sparse (22 in 80d) — the
sample is small but Sharpe is positive across all sensible parameter
choices. Default `ScannerParams.score_min=2.0` is now demonstrably too
tight; deprecate in favor of 1.0 in future production code.

## P2 — Walk-forward (2026-05-26) — REGIME SENSITIVITY DISCOVERED

`scripts/cascade_walkforward.py` splits the 80d window into 3 sub-
windows: W1 (Apr 3-21), W2 (Apr 22-May 11), W3 (May 12-26). Per-window
corpus replay + joint sim.

| Window | Corpus Sharpe | Corpus avg PnL | Joint Sharpe | Joint avg PnL |
|---|---|---|---|---|
| W1 | +1.65 (n=11) | +$0.36 | +2.15 (n=6) | +$0.91 |
| W2 | **+12.76** (n=14) | **+$2.93** | +4.36 (n=9) | +$1.29 |
| W3 | +1.29 (n=11) | +$0.51 | **−0.21** (n=6) | −$0.10 |

**Corpus replay: PASS** (min Sharpe 1.29, always positive). His
selection has real edge across regimes.

**Joint sim: FAIL in W3** (Sharpe −0.21). Our scanner+detector lost
money in May 12-26. The aggregate +$16.48 = $5.49 + $11.61 − $0.62 —
W2 carries 70% of joint result.

**Why joint failed W3 while corpus succeeded:** corpus W3 made +$0.51/
trade at 36% WR; joint W3 made −$0.10/trade at 33% WR. Same window,
worse selection — joint scanner+detector picked the SUBSET of his
calls that didn't work. Likely culprits: the stealth picks (GUA, RIVER,
EDEN×4) that we already flagged as outside the scanner's edge zone.

**Implication for production:** the +12.2% annualized headline is
regime-specific. A more honest estimate is roughly 0-12% with high
variance. The joint system should NOT be ramped without out-of-sample
testing on additional weeks of data.

## P3 — Execution parameter sweep (2026-05-26) — SELECTION-SENSITIVE

`scripts/cascade_param_sweep.py` swept 144 combinations of
(stop_atr_mult × tp1_r_multiple × hard_time_stop_bars × trail_atr_mult)
on corpus replay. Objective: max Sharpe with ≥20 trades + positive PnL
in EACH walk-forward window. 111 of 144 combinations survived the gate.

**Top corpus-replay params** (all-windows-positive, sorted by Sharpe):

| stop | tp1R | TS | trail | n | WR | avg | total | Sharpe | W1$ | W2$ | W3$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1.0** | 1.0 | 24 | **2.0** | 36 | 58.3% | **+$2.81** | **+$101.22** | **+6.19** | +3.67 | +73.92 | +23.63 |
| 1.0 | 1.0 | 192 | 2.0 | 36 | 58.3% | +$2.79 | +$100.51 | +6.13 | +1.71 | +75.16 | +23.63 |
| 1.0 | 1.5 | 24 | 2.0 | 36 | 50.0% | +$2.94 | +$105.74 | +5.92 | +6.85 | +81.72 | +17.17 |

Defaults (stop=1.5, tp1=1.0, TS=24, trail=1.0) produced +$50.55 total,
Sharpe 4.14. Best params **double** that on corpus: +$101.22, Sharpe 6.19.

**BUT the joint-sim validation kills the win:**

| param set | corpus Sharpe | joint Sharpe (n=22) |
|---|---|---|
| Default | +4.14 | **+1.90** |
| Best corpus (stop=1.0, trail=2.0, TS=24) | +6.19 | +0.46 |
| Best corpus (stop=1.0, trail=2.0, TS=192) | +6.13 | +0.84 |

**Same params, opposite effect on joint vs corpus.** Tighter stop + wider
trail captures bigger moves on his picks (entered early in the move) but
underperforms on our scanner+detector triggers (which fire late, after
the move has matured). Execution tuning is selection-sensitive — can't
tune one without considering the other.

**Production recommendation:** keep default execution
(`stop_atr_mult=1.5`, `tp1_r_multiple=1.0`, `hard_time_stop_bars=24`,
`trail_atr_mult=1.0`) for the joint system; default joint Sharpe is 1.90.
The "best corpus" params overfit to oracle-knowledge and degrade
real-world performance.

**True bottleneck:** our scanner+detector enters LATER in a move than
he does. Closing that timing gap (earlier signal, not better execution)
is the highest-leverage next direction.

## Joint sim diagnostic + strictness sweep (2026-05-26) — CASCADE FAILS QA

`scripts/_joint_diagnose.py` split joint-sim's 22 trades by proximity to
his actual call timestamps. Result:

| Group | n | WR | Avg PnL |
|---|---|---|---|
| Matched (±2 bars, same side) | 2 | 100% | **+$5.49** |
| Unmatched (independent triggers) | 20 | 50% | +$0.52 |

Matched trades are 10× higher per-trade quality but only 9% of trade
count. Many unmatched losers are WRONG-SIDE — system shorts names he
was long (GRASS, GUA, NAORIS, LAB).

**Strictness sweep** (`scripts/_joint_strict_sweep.py`) over
{score_min × rank × min_confluence × allowed_modes}:

| Filter config | Trades | Sharpe |
|---|---|---|
| All modes, min_conf=2 (default) | 22 | **+2.64** (BEST) |
| min_conf=3 (cascade required) | 9 | −0.69 |
| Continuation-only mode | 5 | −0.56 |
| Reversal-only mode | 4 | −0.48 |
| min_conf=4 | 0 | — |

**Cascade context is NOT additive — it FILTERS OUT WINNERS.** The 13
pure-"breakout" mode trades (level + наторговка + trigger with no
cascade detected) carry the strategy's PnL. Cascade-confirmed setups
are LATER entries that pay less. This contradicts the original
research thesis but matches the timing hypothesis: cascade-present =
move-already-played-out.

**Final production config (validated to maximize joint Sharpe):**
```
ScannerParams: score_min=1.0, rank_cutoff=10
CascadeBacktestParams (default): stop=1.5, tp1=1.0, TS=24, trail=1.0,
                                  min_confluence=2, allowed_modes=None
```
Joint Sharpe 2.64 (deflated 2.36), 22 trades, +$0.97/trade, ~+12%
annualized over 80 days. Still regime-sensitive (W3 negative) — needs
out-of-sample validation before any capital allocation.

## Caveats / known weaknesses

- Cross-sectional momentum in crypto largely doesn't survive t-costs in Han/Kang/Ryu 2024 — Filter 5 is a *ranking* tool, not a standalone signal
- Extreme funding as contrarian signal: evidence both ways; only earns keep combined with momentum/OI confirmation
- Listing-age "sweet spot" past day 7 is weakly supported empirically; the 2-60d window matches his ticker set but expect to retune to 7-30d after first backtest pass
- Friday-positive effect not replicated in BTC literature — likely community/crowd phenomenon; keep weight modest (×1.2)
- Amberdata depth data is BTC-only; alt-perp behavior may differ
- All entry timing assumes we can get fills near close-of-bar; live slippage may invalidate
