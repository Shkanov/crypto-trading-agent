# Strategy tuning recommendations — synthesis from 5 parallel research streams

Generated 2026-05-27. Inputs: 1-year backtest across 23 symbols showed
indicator-confluence 1/23 positive (−$8.4k agg, Sharpe −6.24), mean-reversion
0/23 positive (−$1.1k), funding-harvest 5/23 positive (+$52, deflated Sharpe
+0.16), cascade joint-sim Sharpe 0.73 / W3-walkforward fail. Research streams:
trend-follow regime filters, mean-reversion validity, funding-harvest
optimisation, cross-sectional alt selection, risk/validation framework.

---

## 0. The honest read

Three of four strategies in the repo are broken at their current calibration,
and one (funding) is barely positive. The literature is unanimous on **why**:
intraday rules without regime gates lose on chop; fixed-notional sizing on
mismatched-vol assets destroys Sharpe; absolute thresholds on a heterogeneous
universe leave most symbols dormant; ATR-style stops on autocorrelated returns
let losers run; 3-window walk-forward + naive deflated-Sharpe under-penalises
the overfit. None of this is a "find a better signal" problem — it's an
infrastructure problem.

The single highest-ROI change available is **realistic cost modelling +
vol-targeted sizing**. Both are independent of strategy choice and typically
cut backtest Sharpe by 30–50% — i.e. they expose which "winners" are actually
real. Do these before any further parameter sweep.

The single highest-ROI **new strategy direction** is **cointegrated pairs
(ETH/BTC, SOL/ETH)** plus **proper cross-sectional funding-carry**. Both have
strong academic evidence and are essentially absent from the current repo.

---

## 1. Cross-cutting changes (apply to every strategy)

Priority order — each cheap to add and high-impact.

### 1.1 Vol-targeted %-of-equity sizing — replace fixed $100 notional

```
target_vol_annual  = 0.25                # per-position vol target
kelly_fraction     = 0.20                # de Prado-style fractional Kelly
realized_vol_30d   = EWMA(returns, lambda=0.94)  # RiskMetrics
notional_i         = kelly_fraction * equity * target_vol_annual / realized_vol_i
cap                = 2.0 * equity        # margin reality
notional_i         = min(notional_i, cap)
```

Add a portfolio-level vol scalar at 15% annualised target using
Ledoit-Wolf shrunk covariance on 60d returns:
`k = min(1.0, 0.15 / realized_portfolio_vol_annualized)`, applied to all
per-instrument notionals after step 1.

**Why:** Carver ch. 10; AQR Hurst/Ooi/Pedersen 2017; Hood-Raughtigan SSRN
4773781 (vol-targeting and trend-following are the same factor). For crypto
fat tails, half-Kelly (0.5×) is the literature recommendation; 0.20× is more
conservative and standard among retail-scale crypto practitioners
(@nope_its_lily, Robot Wealth playbook). Cap at 2× equity reflects Binance
margin reality.

**Expected impact:** +0.2 to +0.5 Sharpe just from removing the noise of
mismatched bet sizes; max DD typically drops 30–50%.

**Where:** `src/services/backtest.py` — replace `$100` with a sizer module
called per-trade. New `src/services/sizing.py` with `VolTargetSizer` class.

---

### 1.2 Realistic cost model

Three components, all currently missing or under-modelled:

```
fee_taker_perp   = 0.0005   # 5 bps
fee_taker_spot   = 0.0010   # 10 bps
fee_maker_perp   = -0.0002  # rebate
fee_maker_spot   = 0.0010   # no rebate

# Almgren-Chriss sqrt-impact
impact_bps = 0.5 * spread_bps + impact_k * sqrt(notional / adv_5m)
impact_k_major = 0.05        # BTC/ETH/SOL
impact_k_mid   = 0.15        # alts

# Non-fill risk for limits
maker_fill_rate_trending = 0.50
maker_fill_rate_chop     = 0.80

# Funding accrual on all perp positions held across funding events
pnl_funding = -side * notional * funding_rate_8h  # for each event crossed
```

**Why:** AlphaArchitect 2019 ("Decay of Anomalies"): median equity strategy
loses 40–60% of paper Sharpe to realistic costs; crypto 5m bars typically
30–50%. The current harness assumes taker fills at mid + tiny spread,
ignores funding accrual on cross-cycle holds, ignores impact entirely.

**Expected impact:** Will likely DROP backtest Sharpe by 30–50%. That's the
point — it's the gap between paper and real that explains your OOS losses.

**Where:** `src/services/backtest.py` — wrap every simulated fill through
a `Costs.apply(...)` function. Add funding-accrual to multi-cycle holds.

---

### 1.3 Walk-forward → CPCV(N=10, k=2) with PBO

Replace the 3-window walk-forward with **Combinatorial Purged
Cross-Validation** (López de Prado, *Advances in ML Finance*, ch. 12):
- 10 time-ordered groups; train on 8, test on 2.
- Run all C(10,2)=45 combinations.
- **Embargo = max feature lookback** between groups.
- Report **PBO (Probability of Backtest Overfitting)** = fraction of folds
  where the in-sample best is below-median OOS. Require PBO < 0.5.

**Why:** Bailey-Borwein-de Prado-Zhu 2017 PBO formula. With 3 windows the
variance of the test statistic is so high a 60% in-sample-positive result
is statistically indistinguishable from noise.

**Where:** New `scripts/cpcv_validate.py`. Replace
`scripts/cascade_walkforward.py` calls with CPCV variant.

---

### 1.4 Exact deflated Sharpe (Bailey & López de Prado 2014)

Current harness uses `sqrt(log(n_trials)) / sqrt(n_trades)` — right family,
wrong constant. Use the published formula:

```python
def deflated_sharpe(sr_obs, sr_trials, n_trades, skew, kurt):
    """Bailey & López de Prado 2014."""
    gamma = 0.5772156649  # Euler-Mascheroni
    N = len(sr_trials)
    v_sr = np.var(sr_trials, ddof=1)
    e_max_sr = sqrt(v_sr) * (
        (1 - gamma) * norm.ppf(1 - 1/N) +
        gamma * norm.ppf(1 - 1/(N * exp(1)))
    )
    denom = sqrt(1 - skew * sr_obs + ((kurt - 1) / 4) * sr_obs**2)
    return norm.cdf((sr_obs - e_max_sr) * sqrt(n_trades - 1) / denom)
```

For N=20 trials, T=100 trades, V[SR]=0.5: a raw SR of 1.5 deflates to ~0.5.
Require DSR > 0.95 at 95% confidence before any strategy graduates from
research to paper-trade.

**Where:** `src/services/backtest.py::_deflated_sharpe` — swap the current
heuristic for the exact formula. Track V[SR] across the parameter grid.

---

### 1.5 Survivorship-bias-corrected universe

Ammann/Burdorf/Liebi/Stöckl SSRN 4287573: equal-weighted survivorship bias
on crypto is **62% per year** — a "buy top-20 by mcap monthly" backtest can
overstate returns by 4×. Binance has delisted 50+ perp markets since 2022.

Fix: maintain a point-in-time delisting log. At each backtest timestamp,
the eligible universe = symbols that were live then. Delisted symbols
auto-exit at last traded price, not zero.

Data sources: Binance announcement API (`/sapi/v1/system/status`,
`/wapi/v3/markets`), or scrape `https://www.binance.com/en/support/announcement`.
Kaiko and CoinAPI also publish point-in-time universes.

**Where:** New `data/research/universe/binance_delistings.json` with
`{symbol: {listed_ms, delisted_ms}}`. New `src/scanners/universe_pit.py`.
Modify all backtest scripts to filter via PIT universe.

**Expected impact:** 20–40% Sharpe haircut on current backtests — but a
TRUE number to optimise against. Without this any 1y backtest is fiction.

---

### 1.6 Drawdown circuit breakers

Three independent circuits:
1. Trailing equity DD: −10% → halve all sizes; −20% → flatten, 14-day cooloff.
2. Realized portfolio vol: > 2× target for 5 consecutive days → halve sizes.
3. Daily PnL: −3% in one day → no new entries until next session.

**Why:** Carver ch. 16; Clenow *Trading Evolved* ch. 11. AHL targets 15%
vol with −20% soft stop. Renaissance does graded de-risking, not binary.

**Where:** `src/services/risk_circuits.py` — new module called pre-trade.

---

## 2. Strategy-by-strategy recommendations

### 2.1 Indicator confluence (`src/strategies/indicator_confluence.py`)

**Diagnosis:** Canonical retail failure pattern — MACD+EMA+RSI confluence on
15m bars with no regime gate, ~700 trades/yr/symbol, drawdowns 30–78%.

**Changes ranked by ROI:**

1. **Regime gate (cheapest, biggest win):**
   - Stack: `ADX(14) on H1 > 25` AND `Choppiness(14) on H1 < 50`.
   - Published precedent: QuantPedia D1H1 study took SR 0.33 → 0.80 just by
     adding a daily-trend filter to a 1h MACD strategy.
   - Expected: trade count −60–70%, win rate +6–10 pp, max DD halves.

2. **Drop the trigger timeframe from 15m to 1H or 4H:**
   - Profitable crypto trend strategies cluster at 4H–daily, ~30–150 trades
     /yr/symbol. 700/yr is edge-decay territory.
   - Han-Kang-Ryu SSRN 4675565: most sub-daily momentum returns collapse to
     insignificance after costs.

3. **Replace MACD/EMA/RSI confluence with a single rule:**
   - `H1 close > Donchian(55).high AND H4 EMA20 > EMA50`.
   - Donchian (Turtle 20/55) has positive returns across 2017 bull, 2018 bear,
     2020 bull, 2022 bear, 2023+ recovery on BTC (RogueQuant Substack).

4. **Replace fixed ATR stop with Chandelier(22, 3.0):**
   - StratBase 2024: Chandelier improves profit factor 26–48% vs fixed
     trailing. 2.5–3× ATR is the validated multiplier.
   - Keep initial 2-ATR hard stop; trail with Chandelier.
   - DO NOT add profit targets — they kill trend expectancy.

5. **Universe cull:**
   - Drop to top-8 majors by 30d ADV: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX.
   - Begušić-Kostanjčar arXiv 1904.00890: momentum is concentrated in
     liquid majors. Alt slippage explains most of the −37% CAGR.

**Expected combined:** From SR −6.24 to SR 0.5–1.0 OOS. Not magic — just
de-noised.

---

### 2.2 Mean reversion (`src/strategies/mean_reversion.py`)

**Diagnosis:** Pure single-instrument MR on directional crypto is near-dead
edge. Your 25–35% win rate is the inverse of what MR is supposed to produce
(textbook MR: 65–80% WR, R:R 1:2). Entries are hitting trend starts, not
overshoot ends.

**Two paths:**

**Path A — Fix the existing strategy (limited upside):**

1. Replace ADX<20 with a 3-test gate: Hurst(100-bar) < 0.4 AND Variance-Ratio
   test rejects RW at p<0.05 AND OU half-life ∈ [4, 30] bars. Expect 60–80%
   of current entries to disappear.
2. Replace BB/RSI trigger with **residual z-score < −2.0 vs BTC** + funding
   at 95th-percentile of trailing 30d.
3. Triple-barrier exits (López de Prado): TP at 0.7× entry deviation, SL at
   2.0×, time stop at 3× OU half-life. The time stop is non-negotiable.
4. Universe screen: H<0.45 on 180d, AC1<−0.05, RV not in top decile, no
   active narrative in 30d. Probably leaves you with stables-pegged and
   wrapped pairs only.

**Path B — Pivot to cointegrated pairs (recommended):**

This is where the academic evidence is **strongly positive**:
- Lintilhac & Tourin 2017 (arXiv 2109.10662): dynamic cointegration on
  BTC-LTC, hourly, Sharpe ~3 after costs.
- Tadi & Witzany 2023 (arXiv 2305.06961): copula-based cointegration on
  hourly bars.
- Fischer-Krauss-Deinert 2019 (MDPI JRFM 12/1/31): residual-MR on 40 coins,
  120-min hold, Sharpe ~2.3 after costs.

**Recipe:**
- Pairs to test: ETH/BTC, SOL/ETH, AVAX/SOL, ETH/stETH, BTC perp/spot basis.
- Cointegration: 60–90d lookback, Engle-Granger or Johansen,
  ADF p < 0.05, refit weekly. Drop pair if p > 0.05 for 3 consecutive refits.
- Z-score: enter at |z| ≥ 2.0, exit at |z| < 0.5, stop at |z| > 3.5.
- Bar: 1H. Round-trip cost budget: 6–12 bps.

**Where:** New `src/strategies/pairs_cointegration.py`. New
`scripts/backtest_pairs.py`.

**Path C — Long-shot fallback: Avellaneda-Stoikov MM bot:**

If both A and B fail, the only "mean-reverting capital" deployment with a
documented stable retail Sharpe > 1 is a vanilla A-S market-maker on
BTC/USDT or ETH/USDT perp. Hummingbot has a reference implementation.
γ ≈ 0.1, k ≈ 1.5, inventory limits ±0.5% of capital. Falces-Marin PMC 9767337
shows RL improvements on top of vanilla A-S.

**Reasonable order:** B first (largest evidence base + new direction), then
A as a low-effort cleanup. C only if both fail.

---

### 2.3 Funding harvest (`src/strategies/funding_harvest.py`)

**Diagnosis:** Only strategy in the suite that earned anything net (+$52),
but 18/23 symbols never triggered. Wrong thresholds, wrong universe.

**Three changes:**

1. **Per-symbol rolling-z thresholds instead of absolute bps:**
   ```
   window         = 60d
   z_entry        = +1.5σ           # ~top decile of symbol's own funding history
   z_exit         = +0.3σ           # near median
   z_stop         = −2.0σ           # regime flipped against us
   ```
   This single change activates the 18 dead symbols and is the highest-leverage
   fix. Expect 5–10× the trade count and (if the per-symbol expected funding
   net of cost is positive) 5–10× the PnL.

2. **Universe expansion to 60–100 mid-cap and recently-listed alt perps:**
   - Filter: listed >30d, spot 24h vol >$5M, spot borrow available,
     rolling 30d std(funding) > exchange median.
   - The cross-section carry premium (Fan-Jiao-Lu-Tong SSRN 4666425) shows
     43.4% p.a., Sharpe 0.74, concentrated in lower-cap higher-OI names.
   - Majors are dominated by Ethena/Resolv — don't compete on BTC/ETH.

3. **Cost-gated entry:**
   ```
   roundtrip_cost_apr = (2*taker + 2*spread + borrow_apr) * cycles_per_year
   min_edge_apr       = 0.05         # 5% p.a. minimum gross edge
   enter if (predicted_funding * 1095) - roundtrip_cost_apr > min_edge_apr
   ```
   Currently the harness has no cost-vs-funding-rate sanity check at entry,
   so small alts with 30 bps roundtrip are entering on signals that can't
   possibly pay back.

**Stretch goals:**
- Tier majors to **sUSDe deposit** (3.72% APY benchmark). Don't run DIY on BTC/ETH.
- Add **cross-sectional carry overlay**: weekly rebalance, long top-3-funding
  perps + short bottom-3-funding perps, 20–30% of book. Direct implementation
  of Fan et al.
- **Cycle timing:** enter T−30min before settlement when predicted funding
  ≥ z-threshold; re-evaluate each cycle on predicted rate.
- **Kill switch:** flatten all positions if aggregate market liquidations
  >$1B in 5min (Oct-10-2025 style) or BTC 1h move >4σ.

**Target:** Sharpe 1.5–2.5 (industry-realistic 2025–2026). The +$52/yr
becomes a credible $10–30 per $100 of capital actively deployed.

**Where:** `src/services/backtest.py::backtest_funding` already accepts
`FundingBacktestParams` — extend with `z_entry`, `z_exit`, `window_days`,
`min_edge_apr`. Add new universe builder.

---

### 2.4 Cascade breakout + scanner (`src/strategies/cascade_breakout.py` +
`src/scanners/aktradescalp_scanner.py`)

**Diagnosis:** Trader earns +180 bps/trade vs random; mechanical scanner
captures Sharpe 0.73. The +110 bps gap is state-conditional execution
that the scanner doesn't replicate.

**Three changes ranked by where the missing alpha most likely lives:**

1. **Microstructure-conditional entry layer (biggest expected win):**
   When the scanner approves a symbol, wait up to 20–40 minutes for one of:
   - CVD slope flips with price (no divergence)
   - Top-5 order-book imbalance >1.5× in trade direction
   - Liquidation flush that reclaims the breakout level

   Skip the trade if none fires inside the window. This mechanises what the
   discretionary trader almost certainly does. Robot Wealth crypto-stat-arb
   feature ideas; arXiv 2208.09968 attention-momentum literature.

   Expected: +50–100 bps/trade — this is where most of the missing alpha lives.

2. **Promote cross-sectional momentum to primary signal; demote vol-z/OI-z
   to confirmers:**
   - Primary rank: 30d/7d cross-sectional return (Drogen-Hoffstein-Otte
     SSRN 4322637).
   - Confirmers (must have ≥2 of): 1h vol-z ≥ 2, OI-z ≥ 2, |funding| > 95th pct.
   - Liu-Tsyvinski 3-factor (market, size, momentum) is the canonical
     anchor — current scanner is underweighting momentum.

   Expected: +0.3–0.5 Sharpe holding execution constant.

3. **Survivorship-bias-corrected backtest universe (see §1.5):**
   Without this, the strictness/cascade sweeps already on disk are
   calibrated to a phantom signal. Expect 20–40% Sharpe haircut, but a
   true number to optimise against.

**Stretch:**
- Add Santiment social-volume DELTA (not level) as tiebreaker among
  top-decile momentum candidates (Liu/Tsyvinski attention factor).
- Drop listing-age as positive selector but keep a 30d microstructure
  safety floor.
- For the channel-mimic specifically: dedup intra-30min duplicate calls
  in the corpus (currently double-counts EDEN×2, UB×2 confirmations).
- Treat "watch/setup" messages distinct from entries — PHA's 06:00 → 15:56
  monitoring proves the 07–12 UTC session is too narrow; either extend or
  implement watchlist persistence through the day.

---

## 3. New strategy directions worth opening

### 3.1 Cointegrated pairs (HIGH PRIORITY)

Detailed in §2.2 Path B. The single highest-evidence direction missing from
the current repo.

### 3.2 Cross-sectional funding carry (MEDIUM)

Detailed in §2.3 stretch goals. Direct implementation of Fan et al. SSRN
4666425.

### 3.3 Avellaneda-Stoikov market-making (LOWER, infrastructure-heavy)

Detailed in §2.2 Path C. Sub-ms infrastructure not currently in the repo;
significant rebuild. Defer unless §2.1–§2.3 all fail.

### 3.4 Multi-strategy portfolio with HRP allocation

Once §2.1–§2.3 each have an independent positive backtest:
- Allocate via Hierarchical Risk Parity (López de Prado 2016) on rolling
  90d strategy returns, monthly rebalance.
- Falls back to inverse-vol if HRP weights are high-turnover.
- Expect portfolio Sharpe ~1.3–1.5× best single-strategy if pairwise
  correlations stay <0.5.
- DeMiguel-Garlappi-Uppal 2009 warning: for ≤5 strategies, equal-weight
  often beats fancy methods OOS. Start equal-weight.

**Where:** New `src/services/portfolio.py` with allocator class.
`src/orchestrator.py` consumes the weights.

---

## 4. Implementation priority list

Ranked by ROI on engineering time. Each item independent; do in order.

| # | Change | Files touched | Expected impact | Time est |
|---|---|---|---|---|
| 1 | Realistic cost model (1.2) | `src/services/backtest.py` | Exposes false-positive strategies; SR haircut 30-50% | 1-2 days |
| 2 | Vol-targeted sizing (1.1) | New `src/services/sizing.py`; `backtest.py` | +0.2-0.5 SR; DD -30-50% | 2-3 days |
| 3 | Funding rolling-z thresholds (2.3.1) | `src/services/backtest.py` | Activates 18 dead symbols; 5-10× funding PnL | 1 day |
| 4 | Funding universe expansion + cost gate (2.3.2-3) | New universe builder | Captures Fan et al. 43.4% premium | 2-3 days |
| 5 | Cointegrated pairs strategy (2.2 Path B) | New `pairs_cointegration.py` | Highest-evidence new direction | 4-5 days |
| 6 | Trend regime gate ADX+CHOP (2.1.1) | `indicator_confluence.py` | -60% trades; +6-10pp WR | 1 day |
| 7 | Trend universe cull + bar to H1/H4 (2.1.2, 2.1.5) | Config | Removes cost-leak | 0.5 day |
| 8 | Donchian + Chandelier replacement (2.1.3-4) | `indicator_confluence.py` | Simpler, more robust | 1-2 days |
| 9 | Survivorship-corrected universe (1.5) | New PIT module | True numbers, -20-40% SR | 3-4 days |
| 10 | CPCV + PBO validation (1.3) | New `cpcv_validate.py` | Robust OOS estimate | 2 days |
| 11 | Exact deflated Sharpe (1.4) | `backtest.py` | Honest reporting | 0.5 day |
| 12 | Drawdown circuits (1.6) | New `risk_circuits.py` | Tail protection | 1 day |
| 13 | Mean-rev → triple-barrier exits (2.2 Path A) | `mean_reversion.py` | Bounded loss tail | 1-2 days |
| 14 | Cascade microstructure entry (2.4.1) | `cascade_breakout.py` | +50-100 bps/trade | 3-4 days |
| 15 | Cascade momentum-as-primary (2.4.2) | `aktradescalp_scanner.py` | +0.3-0.5 SR | 1-2 days |
| 16 | Cross-sectional funding carry overlay (2.3 stretch) | New script | Captures cross-section premium | 2-3 days |
| 17 | HRP multi-strategy allocation (3.4) | New `portfolio.py` | +30-50% combined SR | 2-3 days |

**Suggested ordering: 1 + 2 first (they expose which strategies are real).
Then 3 + 4 (funding is the only positive baseline; lock that in).
Then 5 (new direction with the strongest evidence).
Then 6-8 (trend tune; high probability of going from −SR to 0-1 SR).
Then 9-12 (the validation/risk meta-layer).
Then 13-16 (per-strategy refinements).
Then 17 (combine).**

Total: ~30-40 engineering days for the full programme. The first 5 items
(~10 days) should produce the largest delta to expected live PnL.

---

## 5. Things to STOP doing

- **Stop running 15m intraday MR on directional alts.** The literature is
  unanimous and your 1y backtest confirms it. Pivot to pairs or kill.
- **Stop using ADX<20 alone as MR regime filter.** It's the weakest filter
  available and your 0/23 result shows it doesn't work.
- **Stop running indicator confluence on the full 23-symbol mid-cap
  universe at 15m without a regime gate.** This is the −$8.4k generator.
- **Stop optimising parameters without survivorship-corrected universe.**
  You're tuning to a phantom signal.
- **Stop reporting Sharpe without DSR alongside.** A raw SR of 1.5 with 20
  trials and 100 trades deflates to ~0.5; without that context every
  reported SR overstates by ~0.8-1.0.
- **Stop trying to beat Ethena on majors-funding.** You can't. Tier into
  sUSDe and concentrate your DIY harvest on mid-cap and new-listing alts.

---

## 6. Concrete next-step proposal

If I were resourcing this myself, the next 2-week sprint:

**Week 1:**
- Day 1: Add Costs class + funding accrual to backtest harness (item #1).
- Day 2-3: Add VolTargetSizer module; wire into all backtest scripts (#2).
- Day 4: Re-run 1y backtest of all 4 strategies with realistic costs +
  vol-targeted sizing. Compare against current numbers — likely 2-3 of the
  4 die outright.
- Day 5: Add funding rolling-z thresholds (#3) and cost-gated entry (#4).
  Re-run funding backtest.

**Week 2:**
- Day 1-3: Build cointegrated pairs strategy on ETH/BTC + SOL/ETH (#5).
  Backtest 2y on those pairs.
- Day 4-5: If pairs SR > 1.5 OOS, write up; if not, debug the
  cointegration test.

After 2 weeks you'll know:
- Whether funding can be tuned to a credible SR ~1.5-2.5 (#3-4).
- Whether pairs can produce SR ~1.5-3 on majors (#5).
- Whether trend+MR die honestly under realistic costs (#1-2).

That's the highest-information-density 2 weeks the research can produce.

---

## 7. Bibliography

### Trend-following
- QuantPedia — How to Design a Simple Multi-Timeframe Trend Strategy on Bitcoin
- Han, Kang & Ryu — Time-Series and Cross-Sectional Momentum in Crypto (SSRN 4675565)
- Begušić & Kostanjčar — Momentum and Liquidity in Cryptocurrencies (arXiv 1904.00890)
- Drogen, Hoffstein, Otte — Cross-sectional Momentum in Cryptocurrency Markets (SSRN 4322637)
- Moskowitz, Ooi, Pedersen — Time Series Momentum (SSRN 2089463)
- Hood & Raughtigan — Volatility Targeting Is Trendy (SSRN 4773781)
- Macrosynergy — Hurst exponent for trends and mean reversion
- MDPI Mathematics 12/18/2911 — Anti-Persistent Hurst in Crypto Pairs
- RogueQuant — Backtesting Turtle Trading Across 40 Markets
- StratBase — ATR Trailing Stop & Chandelier Exit Guide
- Concretum — Position Sizing in Trend-Following

### Mean reversion / pairs
- Padysak & Vojtko — Seasonality, Trend-following, MR in Bitcoin (SSRN 4081000)
- Fischer, Krauss & Deinert — Statistical Arbitrage in Crypto (MDPI JRFM 12/1/31)
- Krauss — Statistical Arbitrage Pairs Trading (J. Econ. Surveys 2017)
- Amberdata — Pairs trading: ADF + Hurst + VR test
- Tadi & Witzany — Copula-Based Trading of Cointegrated Crypto Pairs (arXiv 2305.06961)
- Lintilhac & Tourin — Dynamic Cointegration-Based Pairs Trading (arXiv 2109.10662)
- Bertram — Closed-form OU optimal trading (arXiv 2003.10502)
- Robot Wealth — Exploring Mean Reversion and Cointegration
- Hummingbot — Avellaneda & Stoikov Market-Making Strategy
- Falces Marin et al. — RL approach to Avellaneda-Stoikov (PMC 9767337)
- Hudson & Thames — Triple-Barrier Method
- Quantified Strategies — Mean Reversion Backtests

### Funding harvest
- Schmeling, Schrimpf & Todorov — Crypto Carry (BIS WP 1087 / SSRN 4268371)
- Fan, Jiao, Lu, Tong — Risk and Return of Cryptocurrency Carry Trade (SSRN 4666425)
- Cong & He — Tokenomics of Staking (NBER w33640)
- BIS — Crypto carry, market segmentation, price distortions
- Ethena — Funding Risk docs; Q1 2026 Report
- Resolv — The True Delta-Neutral Stablecoin
- Pendle Boros — Cross-Exchange Funding Rate Arbitrage
- BitMEX — XBTUSD Funding Mean Reversion; 2025Q3 Derivatives Report
- MDPI Mathematics 14/2/346 — Two-Tiered Structure of Crypto Funding Markets
- Amberdata — Impact of Crypto Funding Rates
- QuantJourney — Funding Rates: Hidden Cost, Sentiment, Strategy Trigger
- Buildix — Cash and Carry in Crypto 2026

### Cross-sectional & on-chain
- Liu, Tsyvinski — Risks and Returns of Cryptocurrency (NBER w24877)
- Liu, Tsyvinski, Wu — Common Risk Factors in Cryptocurrency (NBER w25882)
- Borri & Shakhnov — Cross-Section of Cryptocurrency Returns (SSRN 3241485)
- Fracassi & Kogan — Pure Momentum in Cryptocurrency Markets (SSRN 4138685)
- Dobrynskaya — Cryptocurrency Momentum and Reversal (SSRN 3913263)
- Cong/Karolyi/Tang/Zhao — Crypto Factor Zoo (.zip)
- Robot Wealth — Quantifying and Combining Crypto Alphas; Stat Arb Features
- Phillips & Gorse — Predicting Altcoin Returns Using Social Media (PMC6279012)
- Bookmap / Kingfisher — CVD, Liquidation Maps
- Ammann/Burdorf/Liebi/Stöckl — Survivorship and Delisting Bias in Crypto (SSRN 4287573)
- Concretum — Building a Survivorship-Bias-Free Crypto Dataset

### Risk / validation
- López de Prado — Advances in Financial Machine Learning
- López de Prado — Machine Learning for Asset Managers
- Bailey & López de Prado — The Deflated Sharpe Ratio (2014)
- Bailey/Borwein/de Prado/Zhu — Probability of Backtest Overfitting (2017)
- Carver — Systematic Trading; Smart Portfolios
- Clenow — Trading Evolved
- Hoffstein — Newfound Research (multiple essays)
- AQR — Hurst/Ooi/Pedersen Managed Futures research
- Asness/Ilmanen/Israel/Moskowitz — Investing with Style (2015)
- Ledoit & Wolf — Shrunk Covariance Estimator (2004)
- DeMiguel/Garlappi/Uppal — Optimal vs Naive Diversification (2009)
- Harvey/Liu/Zhu — Cross-Section of Expected Returns (2016)
- Hamilton — Regime-switching models (1989)
- Daniel & Moskowitz — Momentum Crashes (2016)
