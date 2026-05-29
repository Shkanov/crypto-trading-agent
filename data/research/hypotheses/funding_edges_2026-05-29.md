# Funding-rate edges in crypto perpetual futures — ranked hypothesis memo

**Author:** quant-strategy-researcher subagent
**Date:** 2026-05-29
**Venue:** Binance spot + USDT-M perps
**Mandate:** find funding edges *beyond* (a) validated cross-sectional carry (`funding_carry.py`, CPCV PASS, PIT-corrected SR ~0.81/+20%) and (b) the already-falsified single-symbol z-thresholded funding harvest (`funding_harvest.py`, 0/38 across the liquidity spectrum). Do not re-pitch either. If nothing beats validated carry, say so.

---

## TL;DR — the honest headline

**The biggest finding is bad news for the workhorse, not a new edge.** The most rigorous current cross-section paper (Cao, Luo, Cheng, Dong, *Anatomy of Cryptocurrency Perpetual Futures Returns*, SSRN 6365329, 170 predictors / 63 significant strategies) reports the **carry / basis Sharpe trajectory falling from ~6.45 (2020–23) → 4.06 (2024) → negative in 2025** [vendor-adjacent peer-review, SSRN]. The mechanism is corroborated independently by BitMEX's 2025 perps report: Ethena USDe + Binance BFUSD + sUSDe clones flooded the book with **structural delta-neutral short inventory**, compressing the "risk-free" carry yield to **sub-4% by mid-2025, often below T-bills** [practitioner/exchange, BitMEX]. This is post-publication crowding in real time. So before chasing new funding edges, the live book should treat plain long-leg carry as a **decaying** premium, not a stationary one.

Against that backdrop, three funding-adjacent ideas are worth the cost of a CPCV run. Two are genuinely distinct from validated carry; one is a sharper reformulation of why carry works that *might* pass where plain carry is decaying. None is a slam-dunk. Ranked:

| # | Card | Distinct from carry? | Verdict | Conf |
|---|------|----------------------|---------|------|
| 1 | **Funding-change (Δfunding) cross-sectional spread** — rank on the *change* in funding, not the level | Yes (level vs first-difference are nearly orthogonal given 0.97+ autocorr) | **MARGINAL→PROMISING** | medium |
| 2 | **Carry-with-momentum/basis double-sort** (Cao two-factor: long-basis × price-volume) | Partly (conditions the carry leg; aims to survive where plain carry decays) | **MARGINAL** | medium |
| 3 | **Extreme-funding reversal, OI-confirmed, directional** (not delta-neutral) | Yes (directional spot bet, opposite construction to carry) | **MARGINAL→LIKELY-NOISE** | low-medium |

Everything else I examined (funding momentum as a standalone signal, 8h-settlement-clock drift, stablecoin/borrow-rate linkage, funding-as-pure-spot-predictor) is **LIKELY-NOISE or not separately testable in this harness** — reasons in the "Rejected / parked" section.

---

## Card 1 — Funding-CHANGE cross-sectional spread (Δfunding rank)

**Thesis.** At each weekly rebalance, rank the eligible perp universe by the **change in funding rate over the trailing window** (e.g. last 7d mean funding minus prior 7d mean), not by the level. Long the perps whose funding is *rising fastest*, short those *falling fastest*, dollar-neutral. Payoff: capture the price drift that *accompanies* funding repricing rather than the (now-compressed) static yield differential.

**Economic rationale.** Funding *levels* are extremely persistent — first-order autocorrelation 0.966–0.998, still >0.80 at lag 8 [vendor/peer, MDPI *Two-Tiered Structure of Crypto Funding Rate Markets*]. Validated carry already harvests the level. But the López-de-Prado-style insight is that a near-unit-root signal carries almost no *new* information period-to-period — the surprise is in the **first difference**. A sharp rise in funding marks fresh crowding/leverage demand that has *not yet* fully repriced spot; the drift the carry paper attributes to "the crowd thinning" should be concentrated in names where funding just moved, not names that have sat at a high level for weeks (those are already arbitraged by Ethena-type inventory). This is the reformulation that could plausibly survive the 2025 carry decay: structural short inventory compresses the *level* spread but cannot instantly absorb a *change*.

**Signal definition.**
- Universe: PIT-filtered top-N USDT perps ex-majors via `universe_pit.eligible_universe_at` (reuse the exact carry pipeline; ≥95% coverage gate).
- Signal: `dfund_i = mean(funding_i, last 21 cycles) − mean(funding_i, prior 21 cycles)`. Weekly cadence (21 funding cycles = 1 week), same as carry.
- Trade: long top-N by `dfund`, short bottom-N, equal-weight, `book_pct_per_side` matched to carry (0.10–0.25). top_n ∈ {2,3,5}.
- Hold: 1 week, then re-rank. Same `cycle_pnl` accounting as `funding_carry.py` (price + funding + 2×taker perp).
- Note: this leg **pays funding on both sides** just like carry — the alpha must come from price drift, so the test is whether `price_pnl` beats costs, not whether you collect yield.

**Expected edge.** No clean published number for the *change* sort specifically (this is a reformulation, not a cited strategy — flag that honestly). Source anchor is the level carry at 43.4%/SR 0.74 pre-decay [Fan, Jiao, Lu, Tong, SSRN 4666425, peer-review/SSRN] and the Cao basis factor that "explains all 63 strategies" [SSRN 6365329]. **Skepticism-adjusted:** if the change-sort has any edge it is a *fraction* of the level edge and likely SR 0.3–0.8 gross, before the 2025 decay; net of costs plausibly SR 0.2–0.5. Treat anything above SR 1 in-sample as overfit until CPCV says otherwise.

**Costs & capacity.** Same cost surface as validated carry, which already passed `costs.py`: 2×5bps taker/leg/side + Almgren-Chriss sqrt-impact (k=0.15 mid / 0.30 small) + funding accrual. **Turnover is the killer here** — a change-sort re-ranks more aggressively than a level-sort (the change signal is noisier week-to-week), so expect higher leg churn and more taker crossing. Capacity is mid/small-cap-limited exactly like carry (the level edge concentrated in lower-cap higher-OI alts per Fan et al.). If net edge < ~30bps/week per leg it's dead — costs alone are ~20–40bps round-trip on small-caps.

**Decay & regime.** Less exposed to the Ethena/BFUSD level-compression than plain carry (that's the whole pitch), but more exposed to chop: in regime drift the change signal whipsaws. Expect it to work in trending-leverage regimes (2021-style, 2024 alt season) and bleed in flat funding regimes (mid-2025 sub-4% world). Post-publication risk is *low* only because it isn't published — which also means low prior it's real.

**Failure modes.** (1) The change of a 0.98-autocorr series is mostly noise → signal is dominated by measurement error → CPCV-OOS-mean ≈ 0. (2) Higher turnover means costs eat the thin edge — the cheapest decisive test is to run it *with full costs on from the first pass* (don't be fooled by a gross-positive). (3) Multiple-testing: window length (21 vs 14 vs 42 cycles) is a free parameter — the sweep must be inside the CPCV/PBO family, not chosen post-hoc.

**Validation plan (this harness).**
- Clone `scripts/cpcv_validate_carry.py` → `cpcv_validate_dfunding.py`. Reuse `build_pit_universe` / `universe_pit` and the shared-histories fetch.
- Replace `rank_for_carry` (level sort) with a Δfunding sort; everything downstream (`build_rebalance`, `cycle_pnl`, daily-bucketing) is unchanged.
- Grid: window ∈ {14,21,42 cycles} × top_n ∈ {2,3,5} × book_pct ∈ {0.10,0.25} = 18 configs (matches the 12-config carry family scale).
- CPCV(N=10, k=2)=45 OOS Sharpes per config; family-wide PBO at S=8 (carry used 70 partitions).
- **Pass gate (the repo's battle-tested conjunction):** PBO < 0.5 **AND** CPCV-OOS-mean > 0, on the IS-best config. Benchmark to beat: validated carry's PIT-corrected OOS-mean (+1.42 ± 2.50 at tn2_bk10 per `cpcv_carry_20260527_125455.json`) — Δfunding only earns a slot if it's *additive/diversifying* to carry, not merely positive.
- Cheapest decisive first test: single 365d run, top-30 PIT universe, window=21, top_n=3, book=0.10, **costs on** — if gross `price_pnl` minus costs is negative, stop here.

**Verdict: MARGINAL → PROMISING, medium confidence.** Biggest uncertainty: whether the first-difference of a near-unit-root funding series retains tradeable signal after costs, or is just amplified noise. This is the *single most defensible* "beyond-carry" idea because it has a real mechanism (level is arbitraged, change is not) and reuses 90% of already-validated, already-CPCV-passing carry code — cheapest possible test for a genuinely distinct signal.

---

## Card 2 — Carry conditioned on basis × momentum (Cao two-factor double-sort)

**Thesis.** Run the validated cross-sectional carry, but **double-sort**: within the high-funding (long) and low-funding (short) buckets, keep only names that also agree with the price-volume/momentum factor. The aim is not a new premium but a *sharper carry* that sidesteps the names where the 2025 structural-short inventory has killed the level edge.

**Economic rationale.** Cao et al.'s headline result is that a **two-factor model — log-basis + a price-volume factor — explains ALL 63 significant perp strategies** [SSRN 6365329, SSRN/peer]. That means basis (≈funding) and momentum/volume are the two systematic axes; carry alone loads on one. Fan et al. show carry returns are *not* subsumed by momentum, size, vol, liquidity [SSRN 4666425] — so the factors are distinct and combinable. Conditioning carry on momentum agreement is the textbook "carry+momentum" double-sort that improves carry Sharpe in FX/commodities and, per Cao, has a crypto analog. The decay defense: the names where Ethena-type inventory crushed funding are precisely the *high-funding-but-no-momentum* names (pure yield, no organic long demand) — exactly the ones a momentum filter drops.

**Signal definition.**
- Same carry universe / cadence / `cycle_pnl` as `funding_carry.py`.
- Long bucket: top-N by funding **∩** positive trailing momentum (e.g. 7d or 30d return rank top half). Short bucket: bottom-N by funding **∩** negative momentum.
- If the intersection is too thin (< top_n), shrink the leg rather than reaching down the rank (avoid forcing trades).
- Momentum lookback ∈ {7d, 30d}; carry params inherited from validated config.

**Expected edge.** Source-claimed: carry SR 0.74 (Fan) pre-decay; two-factor "explains all 63 strategies" but Cao reports the *combined* carry SR collapsed to negative by 2025 — so the double-sort is a **decay-mitigation play, not an edge-amplification play**. Skepticism-adjusted: best case it recovers carry from the 2025 negative regime back to modestly positive (SR ~0.3–0.6); realistic case it just trades fewer names at similar SR to plain carry and adds nothing net of the lost breadth. The honest framing: this is "can we keep carry alive," not "new alpha."

**Costs & capacity.** Strictly *lower* turnover and *fewer* trades than plain carry (it's a filter), so costs are not the binding constraint — the binding constraint is **breadth loss**: dropping to the funding∩momentum intersection on a top-30 universe may leave 1–2 names/leg, which raises idiosyncratic variance and can *worsen* Sharpe even if mean return holds. This is the main reason it may not beat plain carry.

**Decay & regime.** Designed for the post-2025 compressed-funding regime; should help most exactly when plain carry is failing. But momentum in crypto is itself t-cost-fragile (Han/Kang/Ryu 2024, SSRN 4675565, already in project memory) — the momentum overlay could import that fragility.

**Failure modes.** (1) Breadth collapse → high variance → Sharpe down despite "cleaner" names. (2) Double-sorting on a top-30 universe is statistically thin; the intersection is small-n and PBO will be hard to interpret. (3) Look-ahead: momentum rank and funding rank must both be strictly as-of the rebalance timestamp (the carry pipeline already enforces PIT; the momentum addition must use prior-window returns only — same HTF-lookahead trap fixed in `fa33949`).

**Validation plan (this harness).**
- Extend `funding_carry.rank_for_carry` with an optional momentum mask (prior-window return rank), gated by a flag (mirror the `use_momentum_primary` opt-in pattern in `aktradescalp_scanner.py`).
- Clone `cpcv_validate_carry.py`; grid: momentum_lookback ∈ {7d,30d,off} × top_n ∈ {3,5} × book ∈ {0.10,0.25} = ~12 configs. Crucially include `off` so CPCV can tell you whether the overlay *adds* over plain carry (the relevant null is **validated carry itself**, not zero).
- Pass gate: PBO < 0.5 AND CPCV-OOS-mean > **plain-carry OOS-mean** (not just > 0). If it doesn't beat plain carry it's not worth the breadth loss — reject.
- Cheapest decisive first test: run the `off` config to reproduce the known carry number, then the 30d-momentum config on the same universe/window; compare OOS-mean directly before running the full grid.

**Verdict: MARGINAL, medium confidence.** Biggest uncertainty: breadth loss on a realistic top-30 universe likely cancels the cleaner-name benefit. Worth one cheap run because (a) it directly addresses the documented 2025 carry decay and (b) the null is well-defined (beat plain carry or die). If the user's strategic question is "keep the carry bot alive as funding compresses," this is the most relevant test in this memo.

---

## Card 3 — Extreme-funding reversal, OI-confirmed, directional spot bet

**Thesis.** When a single name's funding hits a *universe-relative* extreme **and** open interest is elevated/rising (crowded leverage), take a **directional** position *against* the crowd in spot/perp: short the name on extreme-positive-funding + rising-OI (overcrowded longs about to be flushed), long on extreme-negative-funding + rising-OI (capitulation short squeeze). Hold days, exit on funding normalization or OI flush.

**Economic rationale.** This is the *opposite* construction to carry (directional, not delta-neutral) and to the failed harvest (uses OI confirmation + universe-relative extreme, not own-history z-score). The mechanism: extreme funding marks one-sided leverage; combined with high/rising OI it marks a *fragile* book that mean-reverts violently via liquidation cascade. The reversal story is widely asserted [practitioner: Yellow.com, Gate Wiki, multiple]; the *OI-confirmation* refinement ("extreme funding clearly 'wrong' given price + OI surge") is the part with the most practitioner support and a clean mechanism. Real example: BTC funding flipped positive after **84 consecutive negative days**, +14.5% spot during the streak — i.e., persistent crowded shorts preceded upside [practitioner, FXStreet/K33]. The Cao volume/price factor [SSRN 6365329] is the academic cousin (volume-related factor is one of the two systematic axes).

**Signal definition.**
- Universe: liquid perps (top-50 by volume) so OI data is reliable and slippage bounded.
- Entry: |funding| > universe-wide 95th percentile (the STRICT cross-sectional extreme already implemented in `aktradescalp_scanner` `funding_extreme`) **AND** OI z-score > +2 (rising crowding) over a same-hour-of-day 30d baseline.
- Side: fade the crowd (extreme + funding → short; extreme − funding → long). Directional, single-leg perp.
- Exit: funding crosses back inside its 50th percentile, OR OI z drops below 0 (flush done), OR ATR-based stop, OR time-stop ~3 days.
- Sizing: vol-targeted via `sizing.py` (this is directional risk, unlike the neutral carry).

**Expected edge.** No credible *quantified, cost-inclusive, peer-reviewed* number exists — the reversal claim is overwhelmingly practitioner/forum-tier and explicitly threshold-fragile ("no universal funding level reliably signals a reversal" — repeated across sources). Source-asserted "sharpest reversals" is anecdote, not a Sharpe. **Skepticism-adjusted: this is the weakest-evidenced card.** Treat any backtest SR > 1 as almost certainly period-specific (the big cascades are a handful of dated events — Nov 2025 $19B OI wipe, etc.; fitting to them is selection bias). Realistic prior: small directional edge per signal (+50–150bps when it works) but low hit rate and fat left tail (you're standing in front of a cascade that can run further before reverting).

**Costs & capacity.** Directional perp, taker entry/exit (5bps each) + slippage; cheap on liquid names. Capacity is fine (liquid universe). The real cost is **tail risk / drawdown**, not fees — fading a leveraged extreme can be run over before it reverts. Must be tested *with* the repo's risk-circuit DD model in mind.

**Decay & regime.** Regime-defining: works in capitulation/blow-off regimes, bleeds in trending regimes where extreme funding *persists* (funding "can remain elevated longer than traders expect" during strong trends — directly cited). This is a fundamentally regime-timing bet, which the repo has repeatedly found fragile (cascade strategy W3 failure, LevelBreakout falsification).

**Failure modes.** (1) **Selection bias on cascade events** — the cited "evidence" is a few dated mega-liquidations; a backtest will overfit them and PBO should catch it. (2) Look-ahead in OI baseline (Binance only retains ~30d OI history — *known repo constraint*, see cascade memo — so historical OI z-scores beyond 30d are unavailable; this **structurally limits the backtest window** and is the single biggest practical blocker). (3) Fading a trend = unbounded adverse move before reversion → fat left tail that a Sharpe understates. (4) Threshold-mining: the 95th-pctile + OI-z>2 thresholds are free parameters.

**Validation plan (this harness).**
- Reuse `aktradescalp_scanner`'s universe-relative `funding_extreme` and OI-z machinery; new directional simulator (closest existing analog is `simulate_cascade_breakout` in `backtest.py` — adapt its entry/stop/time-stop loop for a funding-extreme trigger instead of a level break).
- **Window constraint: limited to ~30d of OI history** unless OI is backfilled from an external vendor — so CPCV(N=10,k=2) on 30d is too short for stable folds. *Decisive cheap first test instead:* event-study, not CPCV — collect all universe-relative extreme-funding+OI-surge events in the available window, measure forward 1d/3d returns vs a random-entry baseline on the same names/hours (mirror the cascade corpus-vs-random methodology). Only if the event-study lift is clearly positive and separable do you invest in vendor OI history to enable a real CPCV run.
- Pass gate (event-study stage): forward-return lift over random ≥ +100bps/trade *and* positive in each sub-window (the cascade memo's gate), with explicit left-tail reporting. CPCV gate deferred until OI history allows ≥365d.

**Verdict: MARGINAL → LIKELY-NOISE, low-medium confidence.** Biggest uncertainty: whether there's any edge left after (a) the 30d OI-history wall forces a tiny sample and (b) selection bias on a handful of cascade events is removed. The mechanism is real and the construction is genuinely distinct from carry, but the evidence tier is poor and the harness can't cleanly validate it without external OI data. **Test last, and only via event-study first** — do not build the full simulator before the cheap event-study clears.

---

## Rejected / parked (with reasons — these are NOT worth a CPCV run)

- **Funding momentum as a standalone signal (predict next funding from current funding).** The Inan DAR result [SSRN 5576424] and the MDPI two-tiered paper show funding is ~0.97–0.998 autocorrelated and *predictable as a level* — but that's predicting **funding, not price**. A 0.98-autocorr series is trivially "predictable" (tomorrow ≈ today) and carries no *tradeable* edge on its own; the level edge is already captured by validated carry. **LIKELY-NOISE for trading.** (The *change* of this series is Card 1 — that's the only useful angle.)

- **8h-settlement-clock intraday drift** ("flatten before settlement, rates highest after"). All support is practitioner/blog-tier (metamask, zipmex, bitget); the claim is about *rate* timing and fee-avoidance, not a documented *price* drift you can harvest. No peer-reviewed price-drift-around-funding-timestamp result surfaced. The repo's bar granularity (1h/8h closes) can't cleanly resolve sub-8h drift, and any such edge is sub-bps and crowded by every settlement-aware MM. **LIKELY-NOISE; not testable in this harness.**

- **Stablecoin / USDC-borrow-rate linkage as a funding predictor.** Real linkage exists (Gorton-Klee: Tether de-peg coincided with funding-premia collapse; funding tracks leverage demand) [peer/working-paper], but it's a *contemporaneous co-movement and a tail/crisis indicator*, not a lead-lag tradeable signal — and the repo has no stablecoin-rate data feed. **Parked: no data path, no demonstrated lead.**

- **Funding as a pure cross-sectional spot-return predictor (long-only tilt).** This is just the long leg of carry without the short hedge — strictly worse risk profile, dominated by validated carry, and exposed to the same 2025 decay. **Subsumed by carry; reject.**

- **Re-pitching plain carry or the z-thresholded single-symbol harvest.** Out of scope by mandate; carry is validated-but-decaying, harvest is 0/38 dead. Not re-proposed.

---

## Does anything beat validated carry?

**Net answer: not clearly, today.** No funding edge surfaced that is both well-evidenced and likely to out-Sharpe validated carry on a clean CPCV run. The most important takeaway is the *threat to* carry: independent academic (Cao SSRN 6365329) and practitioner (BitMEX 2025) sources both document carry/basis Sharpe collapsing to negative in 2025 via Ethena/BFUSD structural-short crowding. That makes **Card 1 (Δfunding)** and **Card 2 (carry × momentum)** the priorities — not because they're proven new alpha, but because they are the two cheapest, mechanism-grounded attempts to *replace or rescue* a decaying workhorse, and both reuse the already-CPCV-passing carry pipeline. Card 3 is a distinct directional idea with a real mechanism but poor evidence and a hard data constraint (30d OI history) — event-study it before building anything.

**Recommended order of work:** Card 1 first (cheapest, most distinct, best mechanism) → Card 2 (defends the existing book against documented decay; null = beat plain carry) → Card 3 only via event-study, last.

---

## Sources (tagged by tier)

**Peer-reviewed / SSRN / arXiv (highest weight):**
- Cao, Luo, Cheng, Dong — *Anatomy of Cryptocurrency Perpetual Futures Returns* (170 predictors, 63 sig strategies, two-factor log-basis + price-volume, carry SR 6.45→4.06→negative): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6365329 ; Edinburgh mirror: https://www.research.ed.ac.uk/en/publications/anatomy-of-cryptocurrency-perpetual-futures-returns/
- Fan, Jiao, Lu, Tong — *The Risk and Return of Cryptocurrency Carry Trade* (carry 43.4%/SR 0.74, not subsumed by momentum/size/vol/liquidity): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4666425
- Inan — *Predictability of Funding Rates* (DAR, funding LEVEL predictable, BTC/ETH funding corr 0.84): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424
- *The Two-Tiered Structure of Cryptocurrency Funding Rate Markets* (funding autocorr 0.966–0.998, >0.80 at lag 8): https://www.mdpi.com/2227-7390/14/2/346
- Han, Kang, Ryu 2024 — t-cost caveat on crypto cross-sectional momentum (already in project memory): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565
- Gorton, Klee — *Leverage and Stablecoin Pegs* (Tether de-peg ↔ funding-premia collapse): https://cowles.yale.edu/sites/default/files/2023-04/Leverage%20and%20Stablecoin%20Pegs_April%202023.pdf

**Vendor / exchange research (medium weight):**
- BitMEX — *State of Crypto Perps 2025* (Ethena/BFUSD/sUSDe structural-short inventory; carry yield compressed sub-4%, below T-bills): https://www.bitmex.com/blog/state-of-crypto-perps-2025
- BitMEX — *2025 Q3 Derivatives Report* (funding floor/ceiling structure): https://www.bitmex.com/blog/2025q3-derivatives-report
- Amberdata — *The Impact of Crypto Funding Rates* (qualitative sentiment only; no quantified predictor): https://blog.amberdata.io/the-impact-of-crypto-funding-rates
- Amberdata — *Ultimate Guide to Funding Rate Arbitrage*: https://blog.amberdata.io/the-ultimate-guide-to-funding-rate-arbitrage-amberdata

**Practitioner / forum (idea-generation only, lowest weight):**
- Yellow.com — *How Funding Rates Predict Crypto's Most Violent Reversals* (reversal claim, threshold-fragile): https://yellow.com/learn/how-to-read-funding-rates-crypto-reversals
- Gate Wiki — OI + funding divergence signals: https://www.gate.com/crypto-wiki/article/how-to-interpret-crypto-derivatives-market-signals-funding-rates-open-interest-and-liquidation-data-explained-20251227
- FXStreet/K33 — BTC funding flipped positive after 84 negative days, +14.5% spot during streak: https://www.fxstreet.com/cryptocurrencies/news/bitcoin-funding-rate-flips-positive-following-record-negative-streak-signals-potential-breakout-k33-202605270222
- zipmex — settlement-clock/funding-timing guide (settlement timing, not price drift): https://zipmex.com/blog/how-to-analyze-funding-rates-in-crypto/

**In-repo evidence relied on:**
- `data/research/strategy_tuning/cpcv_carry_20260527_125455.json` — carry PIT-corrected CPCV PASS (IS-best tn2_bk10 OOS +1.42 ± 2.50, PBO 0.114).
- `cpcv_funding_multi_*` (132716 / midcap / smallcap) — single-symbol harvest 0/38 PASS (the falsified variant).
- `src/strategies/funding_carry.py`, `funding_harvest.py`, `src/services/cpcv.py`, `costs.py` — pipeline + cost model + CPCV/PBO API the validation plans target.
